# Repository Guidelines

## Project Structure & Module Organization

This repository is a research fork of OSX for one-stage SMPL-X whole-body mesh recovery. Core training and evaluation entry points live in `main/` (`train.py`, `test.py`, model variants, and `config.py`). Shared networks, losses, preprocessing, visualization, and SMPL-X helpers live in `common/`. Dataset loader classes are under `data/<DATASET>/<DATASET>.py`; images and annotations are expected under `dataset/`. Demo inference code is in `demo/`, conversion utilities are in `tool/`, pretrained checkpoints belong in `pretrained_models/`, and experiment logs/checkpoints/results go under `output/`.

## Build, Test, and Development Commands

- `bash install.sh`: install the CUDA 12.1 / PyTorch 2.1 oriented environment and build vendored `main/transformer_utils`.
- `cd main && bash train.sh`: run the current training configuration.
- `cd main && bash test.sh`: run the current evaluation/debug configuration.
- `cd demo && python demo.py --gpu 0 --img_path input.png --output_folder output --decoder_setting pytorch --pretrained_model_path ../pretrained_models/osx_l.pth.tar`: run single-image inference.

Most training and testing paths are relative to `main/`; run those commands from that directory.

## Coding Style & Naming Conventions

Use Python with 4-space indentation. Follow existing naming: `snake_case` for functions, variables, and files; `PascalCase` for classes such as dataset loaders and model modules; lowercase experiment names such as `output/dcnv4_hand_face`. Prefer helpers in `common/utils/` and the global `cfg` conventions in `main/config.py` over parallel configuration paths. Keep comments brief and preserve nearby context when editing.

## Testing Guidelines

There is no standalone unit-test suite or coverage target. Validate changes with the narrowest relevant executable: `cd main && bash test.sh` for model/evaluation changes, `cd demo && python demo.py ...` for inference changes, and a short `cd main && bash train.sh` or direct `python train.py ...` run for training-loop changes. Note that some evaluation code is configured for partial/debug runs; document any intentional limits in PRs.

## Commit & Pull Request Guidelines

Git history uses short, lowercase, imperative-style summaries such as `update test and demo infer` and `train lr set more small`; keep commits focused and similarly concise. Pull requests should describe the changed model path or dataset flow, list exact commands run, mention required checkpoints or dataset files, and include screenshots or sample outputs for demo/visualization changes.

## Security & Configuration Tips

Do not commit datasets, human model files, pretrained checkpoints, generated meshes, logs, or `output/` artifacts. Required local assets include `common/utils/human_model_files/`, `pretrained_models/`, `dataset/`, and compiled DCNv4/deformable-attention dependencies.
