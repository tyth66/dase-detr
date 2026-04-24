# DASE-DETR Notes

## Active Task
- Dataset: `VisDrone`
- Main config: `configs/dase_detr/dase_dfine_n.yml`
- Current focus: small-object detection with a lightweight DASE encoder

## Common Commands
```bash
python train.py -c configs/dase_detr/dase_dfine_n.yml --device cuda
```

```bash
python train.py -c configs/dase_detr/dase_dfine_n.yml --resume path/to/last.pth --test-only
```

```bash
python tools/inference/torch_inf.py -i DATA/visdrone/test2017/268.jpg -c configs/dase_detr/dase_dfine_n.yml -r runs/DASE_DETR/dfine/train_001/best_stg2.pth
```

```bash
python tools/benchmark/get_info.py -c configs/dase_detr/dase_dfine_n.yml
```

## Recent DASE Changes
- Fixed `tools/benchmark/get_info.py` so invalid `OMP_NUM_THREADS` / `MKL_NUM_THREADS` values are normalized before importing PyTorch.
- Fixed `engine/backbone/hgnetv2.py` so benchmark/model construction works without an initialized distributed process group.
- Compressed `MultiSaliencyScorer` in `engine/deim/common.py` from per-channel dynamic kernels to grouped shared dynamic kernels for the saliency branch.
- Simplified `DynamicSpatialRefine` by removing `SpatialAttention` and replacing the previous channel attention block with SE.
- Compressed `SparseTokenAttention.ffn` by capping the hidden expansion at `2 * dim`.
- Refined sparse recovery in `engine/deim/DASE_Encoder.py` by changing window-level importance to patch-level importance and multiplying it with a distance prior during `Sparse_Window_Aggregation`.

## Current Encoder Notes
- `use_encoder_idx=[0, 2]`
- `input_size=[80, 40, 20]`
- `k_ratio=[0.25, None, 0.75]`
- `win_size=[5, None, 3]`
- For `VisDrone`, the `80x80` branch is treated as the main small-object branch and should be changed conservatively.

## Latest Complexity Result
Command:
```bash
python tools/benchmark/get_info.py -c configs/dase_detr/dase_dfine_n.yml
```

Output summary:
- `Params: 4.6854 M`
- `MACs: 8.7002 G`
- `FLOPs: 17.5252 G`

## Follow-up Priorities
- Validate the current lightweight encoder on `VisDrone` before further compression.
- If more compression is needed, prefer reducing low-resolution branch redundancy before weakening the `80x80` branch.
- Re-check whether patch-level importance improves `AP_s` / recall enough to justify its runtime cost.
