import torch
import torch.nn as nn
import torch.nn.functional as F


def _resolve_dynamic_groups(dim, dynamic_groups):
    if dynamic_groups is None:
        dynamic_groups = max(1, dim // 64)
    dynamic_groups = max(1, min(dynamic_groups, dim))
    while dim % dynamic_groups != 0 and dynamic_groups > 1:
        dynamic_groups -= 1
    return dynamic_groups


class DynamicDWConv(nn.Module):
    def __init__(self, dim, win_size):
        super().__init__()
        self.dim = dim
        self.win_size = win_size
        self.padding = win_size // 2
        self.dw_conv = nn.Conv2d(dim, dim, kernel_size=win_size, padding=self.padding, groups=dim, bias=False)
        self.pw_conv = nn.Conv2d(dim, dim, kernel_size=1, bias=False)

        self.dynamic_kernel = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(dim, dim//4, 1),
            nn.GELU(),
            nn.Conv2d(dim//4, dim * win_size**2, 1))
        
    def forward(self, x):
        B, C, H, W = x.shape
        base_weights = self.dw_conv.weight.unsqueeze(0)  # [1, C, 1, k, k]
        dynamic_weights = self.dynamic_kernel(x).view(B, C, 1, self.win_size, self.win_size)
        fused_weights = base_weights + dynamic_weights
        x = x.reshape(1, B * C, H, W)
        fused_weights = fused_weights.reshape(B * C, 1, self.win_size, self.win_size)
        out = F.conv2d(x, fused_weights, padding=self.padding, groups=B * C)
        out = out.view(B, C, H, W)
        return self.pw_conv(out)


class GroupDynamicDWConv(nn.Module):
    def __init__(self, dim, win_size, reduction=4, dynamic_groups=None):
        super().__init__()
        self.dim = dim
        self.win_size = win_size
        self.padding = win_size // 2
        self.dynamic_groups = _resolve_dynamic_groups(dim, dynamic_groups)
        self.channels_per_group = dim // self.dynamic_groups

        self.dw_conv = nn.Conv2d(dim, dim, kernel_size=win_size, padding=self.padding, groups=dim, bias=False)
        hidden_dim = max(1, dim // reduction)
        self.dynamic_kernel = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(dim, hidden_dim, 1),
            nn.GELU(),
            nn.Conv2d(hidden_dim, self.dynamic_groups * win_size**2, 1),
        )
        self.channel_gate = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(dim, hidden_dim, 1),
            nn.GELU(),
            nn.Conv2d(hidden_dim, dim, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        B, C, H, W = x.shape
        base_weights = self.dw_conv.weight.view(1, self.dynamic_groups, self.channels_per_group, 1, self.win_size, self.win_size)
        dynamic_weights = self.dynamic_kernel(x).view(B, self.dynamic_groups, 1, 1, self.win_size, self.win_size)
        fused_weights = (base_weights + dynamic_weights).reshape(B * C, 1, self.win_size, self.win_size)
        out = F.conv2d(x.reshape(1, B * C, H, W), fused_weights, padding=self.padding, groups=B * C)
        out = out.view(B, C, H, W)
        return out * self.channel_gate(x)


class MultiSaliencyScorer(nn.Module):
    "Multiscale significance scoring network"
    def __init__(self, in_channels, kernel_sizes=[3, 5, 7], reduction=4, dynamic_groups=None):
        super().__init__()
        self.in_channels = in_channels
        self.kernel_sizes = kernel_sizes
        self.num_scales = len(kernel_sizes)

        self.branches = nn.ModuleList(
            [GroupDynamicDWConv(in_channels, k, reduction=reduction, dynamic_groups=dynamic_groups) for k in kernel_sizes]
        )
        self.fuse_attn = nn.Sequential(
            nn.Conv2d(in_channels * self.num_scales, in_channels // reduction, 1),
            nn.GELU(),
            nn.Conv2d(in_channels // reduction, self.num_scales, 1),
            nn.AdaptiveAvgPool2d(1),
            nn.Softmax(dim=1))
        self.channel_proj = nn.Conv2d(in_channels, 1, 1, bias=False)

    def forward(self, x):
        B, C, H, W = x.shape
        feats = torch.cat([branch(x) for branch in self.branches], dim=1)  # [B, C * num_scales, H, W]
        weights = self.fuse_attn(feats)  # [B, num_scales, 1, 1]
        weights = weights.view(B, self.num_scales, 1, 1, 1)  # [B, num_scales, 1, 1, 1]
        feats = feats.view(B, self.num_scales, C, H, W)  # [B, num_scales, C, H, W]
        fused = (feats * weights).sum(dim=1)  # [B, C, H, W]
        saliency_logits = self.channel_proj(fused).flatten(1)  # [B, H * W]
        saliency_map = F.softmax(saliency_logits, dim=-1).view(B, H, W)
        return saliency_map


class SqueezeExcite(nn.Module):
    def __init__(self, dim, reduction=4):
        super().__init__()
        hidden_dim = max(1, dim // reduction)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Conv2d(dim, hidden_dim, 1),
            nn.GELU(),
            nn.Conv2d(hidden_dim, dim, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return self.fc(self.avg_pool(x))

class DynamicSpatialRefine(nn.Module):
    def __init__(self, dim, win_size=5, reduction=8):
        super().__init__()
        self.win_size = win_size
        self.P = win_size * win_size
        self.dim = dim
        
        self.dw_conv = DynamicDWConv(dim, win_size)
        self.se = SqueezeExcite(dim, reduction)
        self.proj = nn.Sequential(
            nn.Conv2d(dim, dim, kernel_size=1),
            nn.GELU())
        
        self.alpha = nn.Parameter(torch.tensor(0.1)) 
        self.group_conv = nn.Conv2d(dim, dim, 1, groups=min(8, dim//16))
    
    def forward(self, x):
        B, K, P, C = x.shape
        win_size = int(P ** 0.5)
        shortcut = x
        x = x.view(B * K, win_size, win_size, C).permute(0, 3, 1, 2)
        feat = self.dw_conv(x)
        feat = self.group_conv(feat)
        feat = feat * self.se(feat)
        out = self.proj(feat)
        out = out.permute(0, 2, 3, 1).reshape(B, K, P, C)

        return shortcut + self.alpha * out
    
class SparseTokenAttention(nn.Module):
    def __init__(self, dim, dim_feedforward, heads=4, dropout=0.0, max_hw=(40, 40)):
        super().__init__()
        self.dim = dim
        self.heads = heads
        self.scale = (dim // heads) ** -0.5
        ffn_hidden_dim = min(dim_feedforward, dim * 2)

        self.qkv_proj = nn.Linear(dim, dim * 3)
        self.out_proj = nn.Linear(dim, dim)
        self.norm1 = nn.LayerNorm(dim)
        self.ffn = nn.Sequential(
            nn.Linear(dim, ffn_hidden_dim),
            nn.GELU(),
            nn.Linear(ffn_hidden_dim, dim))
        self.norm2 = nn.LayerNorm(dim)
        self.dropout = nn.Dropout(dropout)

        H, W = max_hw
        pos_embed = self.build_sincos_pos_embed(H, W, dim)  # [H, W, dim]
        self.register_buffer('sincos_pos_embed', pos_embed)  
        
    def build_sincos_pos_embed(self, H, W, dim):
        grid_y, grid_x = torch.meshgrid(torch.arange(H), torch.arange(W), indexing='ij')
        grid = torch.stack([grid_y, grid_x], dim=-1).float()  # [H, W, 2]
        grid[..., 0] = grid[..., 0] / H  # y normalized
        grid[..., 1] = grid[..., 1] / W  # x normalized

        dim_each = dim // 4
        dim_t = 10000 ** (2 * (torch.arange(dim_each) // 2) / dim_each)  # [dim//2]

        pos_x = grid[..., 1].unsqueeze(-1) / dim_t  # [H, W, dim//2]
        pos_y = grid[..., 0].unsqueeze(-1) / dim_t

        pos_x = torch.stack([pos_x.sin(), pos_x.cos()], dim=-1).flatten(-2)  # [H, W, dim//2]
        pos_y = torch.stack([pos_y.sin(), pos_y.cos()], dim=-1).flatten(-2)
        return torch.cat([pos_y, pos_x], dim=-1)  # [H, W, dim]

    def forward(self, x, coords=None):
        """
        x: [B, K, 1, C] - token representation
        coords: [B, K, 2] - coordinates of each window center (y, x), int type
        """
        B, K, _, C = x.shape
        x = x.squeeze(2)  # [B, K, C]
        residual = x
        x_norm = self.norm1(x)

        coords = coords.long()
        max_h = self.sincos_pos_embed.shape[0] - 1
        max_w = self.sincos_pos_embed.shape[1] - 1
        h = coords[..., 0].clamp(0, max_h)
        w = coords[..., 1].clamp(0, max_w)
        pos_embed = self.sincos_pos_embed[h, w]  # [B, K, C]
        x_norm = x_norm + pos_embed

        qkv = self.qkv_proj(x_norm).reshape(B, K, 3, self.heads, C // self.heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # [B, heads, K, dh]

        attn_logits = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn_logits.softmax(dim=-1)
        out = (attn @ v).transpose(1, 2).reshape(B, K, C)

        out = self.out_proj(out)
        out = residual + self.dropout(out)
        out = out + self.dropout(self.ffn(self.norm2(out)))
        return out.unsqueeze(2)  # [B, K, 1, C]
