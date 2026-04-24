import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict

from ..core import register
from .common import MultiSaliencyScorer, DynamicSpatialRefine, SparseTokenAttention
from .hybrid_encoder import ConvNormLayer_fuse, CSPLayer, SCDown, VGGBlock, RepNCSPELAN4

__all__ = ['DASE_Encoder']

class TopkSampler(nn.Module):
    def __init__(self, H, W, temperature=0.5, init_std=0.1):
        super().__init__()
        self.H = H
        self.W = W
        self.temperature = temperature
        self.mask_logits = nn.Parameter(torch.randn(1, H, W) * init_std)

    def forward(self, saliency_map, K):
        B, H, W = saliency_map.shape
        mask = self.mask_logits
        if mask.shape[-2:] != (H, W):
            mask = F.interpolate(mask.unsqueeze(1), size=(H, W), mode='bilinear', align_corners=False).squeeze(1)

        logits = saliency_map + mask  # [B, H, W]
        soft_mask = F.softmax(logits.view(B, -1) / self.temperature, dim=-1)  # [B, H*W]
        topk_scores, topk_indices = torch.topk(soft_mask, K, dim=-1)          # [B, K]
        calibrated_map = soft_mask.view(B, H, W)
        return topk_scores, topk_indices, calibrated_map

class AdaptiveSparseWindowExtractor(nn.Module):
    def __init__(self, dim, win_size=5, k_ratio=0.1, max_windows=1280, input_size=(40, 40), gamma=1.0, temperature=0.5, lambda_scale=0.5):
        super().__init__()
        self.win_size = win_size
        self.pad = win_size // 2
        self.max_windows = max_windows
        self.k_ratio = k_ratio
        self.gamma = nn.Parameter(torch.tensor(float(gamma)))
        self.lambda_scale = float(lambda_scale)
        H, W = input_size
        self.topk_sampler = TopkSampler(H, W, temperature)

        self.register_buffer('offsets', self._precompute_offsets(win_size))

    def _precompute_offsets(self, win_size):
        pad = win_size // 2
        y = torch.arange(-pad, pad + 1)
        x = torch.arange(-pad, pad + 1)
        grid_y, grid_x = torch.meshgrid(y, x, indexing='ij')
        return torch.stack([grid_y.flatten(), grid_x.flatten()], dim=-1)  # [P, 2]

    def forward(self, feat_map, saliency_map):
        """
        feat_map: [B, C, H, W]
        saliency_map: [B, H, W]
        """
        B, C, H, W = feat_map.shape
        P = self.win_size * self.win_size
        K = min(int(self.k_ratio * H * W), self.max_windows)
        _, topk_idx, calibrated_map = self.topk_sampler(saliency_map, K)
        topk_coords = torch.stack([topk_idx // W, topk_idx % W], dim=-1)  # [B, K, 2]
        topk_coords[..., 0] = topk_coords[..., 0].clamp(self.pad, H - 1 - self.pad)
        topk_coords[..., 1] = topk_coords[..., 1].clamp(self.pad, W - 1 - self.pad)
        offsets = self.offsets.to(feat_map.device).view(1, 1, P, 2)
        abs_coords = topk_coords.unsqueeze(2) + offsets
        abs_coords[..., 0] = abs_coords[..., 0].clamp(0, H - 1)
        abs_coords[..., 1] = abs_coords[..., 1].clamp(0, W - 1)
        lin_idx = abs_coords[..., 0] * W + abs_coords[..., 1]  # [B, K, P]
        lin_idx = lin_idx.view(B, K * P).long()
        feat_flat = feat_map.permute(0, 2, 3, 1).reshape(B, H * W, C)
        patches_flat = torch.gather(feat_flat, 1, lin_idx.unsqueeze(-1).expand(-1, -1, C))
        patches = patches_flat.view(B, K, P, C)
        score_flat = calibrated_map.view(B, H * W)
        patch_scores = torch.gather(score_flat, 1, lin_idx).view(B, K, P).unsqueeze(-1)
        patch_score_mean = patch_scores.mean(dim=2, keepdim=True)
        patch_mask = torch.sigmoid(self.gamma * (patch_scores - patch_score_mean))
        center = topk_coords.unsqueeze(2).float()  # [B, K, 1, 2]
        points = abs_coords.float()                # [B, K, P, 2]
        dist = torch.norm(points - center, dim=-1, keepdim=True)
        distance_weight = torch.exp(-dist / (self.lambda_scale * self.win_size))
        patch_mask = patch_mask * distance_weight  # [B, K, P, 1]
        patches = patches * patch_mask
        return patches, topk_coords, offsets, calibrated_map

class AttnGateFusion(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.gate = nn.Sequential(
            nn.Linear(dim * 2, dim),
            nn.Sigmoid())

    def forward(self, local_feat, global_feat):
        combined = torch.cat([local_feat, global_feat], dim=-1)
        gate_weights = self.gate(combined)  # [B, K, P, C]
        out = gate_weights * global_feat + (1 - gate_weights) * local_feat
        return out

class DASE(nn.Module):
    def __init__(self, 
                 in_channels=256, 
                 heads=4, 
                 dim_feedforward=1024,
                 dropout=0.0,
                 win_size=5, 
                 k_ratio=0.2, 
                 max_windows=1600,
                 kernel_sizes=[3, 5, 7],
                 input_size=40):
        super().__init__()
        self.win = win_size
        self.input_size = (input_size, input_size)

        self.scorer = MultiSaliencyScorer(in_channels, kernel_sizes)
        self.extractor = AdaptiveSparseWindowExtractor(in_channels, win_size, k_ratio, max_windows, self.input_size)

        self.local_refine= DynamicSpatialRefine(in_channels, win_size)
        self.sparse_attn = SparseTokenAttention(in_channels, dim_feedforward,  heads, dropout, self.input_size)

        self.attn_gate = AttnGateFusion(in_channels)

        self.channel_attn = nn.Sequential(
            nn.Linear(in_channels, in_channels // 4),
            nn.ReLU(inplace=True),
            nn.Linear(in_channels // 4, in_channels),
            nn.Sigmoid())
        self.fusion_linear = nn.Linear(in_channels * 2, in_channels)
        self.patch_importance = nn.Sequential(
            nn.Linear(in_channels, in_channels),
            nn.ReLU(inplace=True),
            nn.Linear(in_channels, 1),
            nn.Sigmoid())
        self.norm = nn.LayerNorm(in_channels)
            
    def forward(self, x):
        B, C, H, W = x.shape
        saliency_map = self.scorer(x)
        identity = x.view(B, C, H * W).permute(0, 2, 1)
        patches, coords, offsets, _ = self.extractor(x, saliency_map)  # [B, K, P, C]
        local_out = self.local_refine(patches)        # [B, K, P, C]
        mean_pool = local_out.mean(dim=2)
        max_pool, _ = local_out.max(dim=2)
        attn_weights = self.channel_attn(mean_pool)
        weighted_pool = (local_out * attn_weights.unsqueeze(2)).sum(dim=2)
        concat = torch.cat([max_pool, weighted_pool], dim=-1)
        sparse_token = self.fusion_linear(concat).unsqueeze(2)
        sparse_out = self.sparse_attn(sparse_token, coords)   # [B, K, 1, C]
        sparse_out = sparse_out.expand(-1, -1, patches.shape[2], -1)  # [B, K, P, C]
        fused_out = self.attn_gate(local_out, sparse_out)
        importance = self.patch_importance(fused_out)
        out = self.Sparse_Window_Aggregation(fused_out, coords, importance, offsets, H, W)
        out = self.norm(out + identity)
        return out
    
    
    def Sparse_Window_Aggregation(self, patches, coords, importance, offsets, H, W):
        B, K, P, C = patches.shape
        device = patches.device
        abs_coords = coords.unsqueeze(2) + offsets.view(1, 1, P, 2)
        abs_coords[..., 0] = abs_coords[..., 0].clamp(0, H - 1)
        abs_coords[..., 1] = abs_coords[..., 1].clamp(0, W - 1)
        center = coords.unsqueeze(2).float()
        points = abs_coords.float()
        dist = torch.norm(points - center, dim=-1, keepdim=True)
        distance_weight = torch.exp(-dist / max(1.0, self.win * 0.5))
        importance = importance * distance_weight
        linear_idx = abs_coords[..., 0] * W + abs_coords[..., 1]
        linear_idx = linear_idx.reshape(B, -1)
        patches_flat = patches.reshape(B, K * P, C)
        importance_flat = importance.reshape(B, K * P, 1)
        batch_offset = (torch.arange(B, device=device) * (H * W)).view(B, 1)
        linear_idx = linear_idx + batch_offset
        linear_idx = linear_idx.view(-1)
        patches_flat = patches_flat.view(-1, C)
        importance_flat = importance_flat.view(-1, 1)
        feat_sum = torch.zeros(B * H * W, C, device=device, dtype=patches.dtype)
        weight_sum = torch.zeros(B * H * W, 1, device=device, dtype=importance.dtype)
        feat_sum = feat_sum.scatter_add_(0, linear_idx.unsqueeze(-1).expand(-1, C), patches_flat * importance_flat)
        weight_sum = weight_sum.scatter_add_(0, linear_idx.unsqueeze(-1), importance_flat)
        weight_sum = weight_sum.clamp(min=1e-6)
        feat_out = feat_sum / weight_sum
        return feat_out.view(B, H * W, C)


@register()
class DASE_Encoder(nn.Module):
    def __init__(self, 
                 in_channels=[512, 1024, 2048], 
                 feat_strides=[8, 16, 32], 
                 hidden_dim=256, 
                 dim_feedforward = 1024,
                 nhead=8, 
                 dropout=0.0,
                 use_encoder_idx=[0, 2], 
                 k_ratio=[0.25, None, 0.75],
                 win_size=[5, None, 3], 
                 input_size=[80, 40, 20],
                 expansion=1, 
                 depth_mult=1,
                 act='silu', 
                 version='dfine'):
        super().__init__()
        self.in_channels = in_channels
        self.feat_strides = feat_strides
        self.hidden_dim = hidden_dim
        self.use_encoder_idx = use_encoder_idx
        self.out_channels = [hidden_dim for _ in range(len(in_channels))]
        self.out_strides = feat_strides
        self.input_proj = nn.ModuleList()
        for in_channel in in_channels:
            proj = nn.Sequential(OrderedDict([
                    ('conv', nn.Conv2d(in_channel, hidden_dim, kernel_size=1, bias=False)),
                    ('norm', nn.BatchNorm2d(hidden_dim))
                ]))
            self.input_proj.append(proj)
        self.encoder = nn.ModuleList()
        for i in range(len(in_channels)):
            if i in use_encoder_idx:
                encoder = DASE(hidden_dim, nhead, dim_feedforward, dropout, win_size[i], k_ratio[i],input_size=input_size[i])
                self.encoder.append(encoder)
        self.lateral_convs = nn.ModuleList()
        self.fpn_blocks = nn.ModuleList()
        for _ in range(len(in_channels) - 1, 0, -1):
            if version == 'dfine':
                self.lateral_convs.append(ConvNormLayer_fuse(hidden_dim, hidden_dim, 1, 1))
            else:
                self.lateral_convs.append(ConvNormLayer_fuse(hidden_dim, hidden_dim, 1, 1, act=act))
            self.fpn_blocks.append(
                RepNCSPELAN4(hidden_dim * 2, hidden_dim, hidden_dim * 2, round(expansion * hidden_dim // 2), round(3 * depth_mult), act=act) \
                if version == 'dfine' else CSPLayer(hidden_dim * 2, hidden_dim, round(3 * depth_mult), act=act, expansion=expansion, bottletype=VGGBlock)
            )
        self.downsample_convs = nn.ModuleList()
        self.pan_blocks = nn.ModuleList()
        for _ in range(len(in_channels) - 1):
            self.downsample_convs.append(
                nn.Sequential(SCDown(hidden_dim, hidden_dim, 3, 2, act=act)) \
                if version == 'dfine' else ConvNormLayer_fuse(hidden_dim, hidden_dim, 3, 2, act=act)
            )
            self.pan_blocks.append(
                RepNCSPELAN4(hidden_dim * 2, hidden_dim, hidden_dim * 2, round(expansion * hidden_dim // 2), round(3 * depth_mult), act=act) \
                if version == 'dfine' else CSPLayer(hidden_dim * 2, hidden_dim, round(3 * depth_mult), act=act, expansion=expansion, bottletype=VGGBlock)
            )    

    def forward(self, feats):
        assert len(feats) == len(self.in_channels)
        proj_feats = [self.input_proj[i](feat) for i, feat in enumerate(feats)]
        for i, enc_ind in enumerate(self.use_encoder_idx):
            H, W = proj_feats[enc_ind].shape[2:]
            src_flatten = proj_feats[enc_ind]
            memory: torch.Tensor = self.encoder[i](src_flatten)
            proj_feats[enc_ind] = memory.permute(0, 2, 1).reshape(-1, self.hidden_dim, H, W).contiguous() 
        inner_outs = [proj_feats[-1]]
        for idx in range(len(self.in_channels) - 1, 0, -1):
            feat_height = inner_outs[0]
            feat_low = proj_feats[idx - 1]
            feat_height = self.lateral_convs[len(self.in_channels) - 1 - idx](feat_height)
            inner_outs[0] = feat_height
            upsample_feat = F.interpolate(feat_height, scale_factor=2., mode='nearest')
            inner_out = self.fpn_blocks[len(self.in_channels)-1-idx](torch.concat([upsample_feat, feat_low], dim=1))
            inner_outs.insert(0, inner_out)
        outs = [inner_outs[0]]
        for idx in range(len(self.in_channels) - 1):
            feat_low = outs[-1]
            feat_height = inner_outs[idx + 1]
            downsample_feat = self.downsample_convs[idx](feat_low)
            out = self.pan_blocks[idx](torch.concat([downsample_feat, feat_height], dim=1))
            outs.append(out)
        return outs
