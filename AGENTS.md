# Repository Guidelines

## Project Structure & Module Organization
`train.py` is the main training entrypoint. Core model, data, optimization, and runtime code lives under `engine/`:
- `engine/backbone/`: backbone definitions and wrappers
- `engine/deim/`: DASE/DEIM decoder, encoder, matcher, and criterion logic
- `engine/data/`: datasets, transforms, and dataloaders
- `engine/solver/`: training and evaluation loops
- `engine/core/` and `engine/misc/`: config loading, registry utilities, logging, and distributed helpers

Experiment configs are stored in `configs/`, with reusable bases in `configs/base/`, dataset settings in `configs/dataset/`, and model variants in `configs/dase_detr/`. Utility scripts live in `tools/` for inference, deployment, benchmarking, dataset preprocessing, and visualization.

## Build, Test, and Development Commands
Create a Python environment and install the packages needed for your workflow before running commands.

```bash
python train.py -c configs/dase_detr/dase_rtdetr_n.yml --device cuda
```
Starts training from a YAML config.

```bash
python train.py -c configs/dase_detr/dase_rtdetr_n.yml --resume path/to/last.pth --test-only
```
Runs evaluation only.

```bash
python tools/inference/torch_inf.py -c configs/dase_detr/dase_rtdetr_n.yml -r path/to/model.pth -i demo.jpg -d cuda
```
Runs PyTorch inference on an image or video.

```bash
python tools/deployment/export_onnx.py -c configs/dase_detr/dase_rtdetr_n.yml -r path/to/model.pth
```
Exports a checkpoint to ONNX.

## Coding Style & Naming Conventions
Follow existing Python style: 4-space indentation, `snake_case` for functions and variables, `PascalCase` for classes, and concise module-level docstrings where already established. Keep config filenames descriptive and suffix them with `_n.yml`, `_s.yml`, etc. Reuse the registry/config patterns in `engine/core/` instead of adding new entrypoints ad hoc.

## Testing Guidelines
This repository does not include a formal `tests/` suite yet. Treat lightweight execution checks as the minimum bar:
- run `train.py --test-only` with the target config after model changes
- run the relevant inference or export script for tool changes
- keep one change scoped to one config/model path when possible

If you add tests, place them near the affected module or create a top-level `tests/` directory and name files `test_*.py`.

## Commit & Pull Request Guidelines
Current history is minimal and starts with `Initial commit: DASE-DETR project`. Use short, imperative commit subjects that explain intent, for example: `Fix VisDrone evaluator shape handling`. In pull requests, include:
- a short problem/solution summary
- affected configs or scripts
- validation performed
- sample output paths or screenshots when changing inference or visualization behavior

## Configuration & Environment Tips
Many scripts assume GPU execution, CUDA-capable PyTorch, and optional extras such as `onnxruntime`, `tensorrt`, `pycuda`, or OpenCV. Keep dataset paths and runtime overrides in YAML or CLI flags instead of hardcoding machine-specific paths.
