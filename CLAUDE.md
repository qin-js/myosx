# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

`myosx` is a **research fork of OSX** (One-Stage 3D Whole-Body Mesh Recovery with Component Aware Transformer, CVPR 2023), which is itself built on Hand4Whole and uses `mmpose`/`mmcv`. Given a single RGB image of a person, the network regresses **SMPL-X** whole-body parameters (body pose, hand poses, jaw pose, face expression, shape) and produces a 3D mesh in one stage.

The fork's actual research work lives in the `pytorch` decoder path (`main/model_core.py` + `common/nets/{vit,my_decoder,DCNv4}.py`): the original MMCV components are being replaced with from-scratch PyTorch modules, with a **frozen pretrained backbone** and only the hand/face heads being trained. Most custom code is commented in Chinese.

## Environment

`bash install.sh` (run from repo root) targets **CUDA 12.1 / torch 2.1.0** and builds the vendored `mmpose` from `main/transformer_utils` via `python setup.py install`. The README's cu113 instructions are the upstream OSX defaults; `install.sh` is the current setup.

Beyond pip deps, the following are required and **not** committed (all gitignored):
- `pretrained_models/` — original OSX checkpoints (`osx_l.pth.tar`, `osx_vit_l.pth`, …). The `pytorch` path loads `osx_l.pth.tar` to populate its frozen backbone.
- `common/utils/human_model_files/` — SMPL/SMPL-X/MANO/FLAME model files (see README §4 for download links). Required at import time by `common/utils/human_models.py`.
- `dataset/` — soft links to images/annotations per dataset.
- `output/` — experiment outputs (logs, `model_dump/` checkpoints, `result/`, `vis/`).

Compiled deformable-attention ops are needed for the custom decoder: `common/nets/my_decoder.py` imports `DCNv4.FlashDeformAttn` and falls back to Deformable-DETR's `ops.modules.MSDeformAttn`; `common/nets/DCNv4.py` imports `DCNv4`. These come from the external DCNv4 package and must be compiled separately.

## Commands

All training/testing runs from the **`main/`** directory (paths like `../pretrained_models` are relative to it). The demo runs from **`demo/`**. `main/train.sh` and `main/test.sh` hold the current working invocations.

```bash
# Train (current dev config: pytorch decoder path, resume from a snapshot)
cd main && python train.py \
    --gpu_ids 0 --lr 2e-4 --lr_mult 0.1 --train_batch_size 64 --num_thread 8 \
    --end_epoch 50 --exp_name output/dcnv4_hand_face \
    --pretrained_model_path ../pretrained_models/osx_l.pth.tar \
    --encoder_setting osx_l --decoder_setting pytorch --grad_clip 1.0 \
    --continue_train --continue_train_path ../output/dcnv4_hand_face/model_dump/snapshot_47.pth.tar

# Test (must use the same --decoder_setting the checkpoint was trained with)
cd main && python test.py --gpu 0 --exp_name output/test_setting1 \
    --pretrained_model_path ../pretrained_models/osx_l.pth.tar \
    --testset HAA500 --decoder_setting pytorch \
    --continue_train_path ../output/dcnv4_hand_face/model_dump/snapshot_49.pth.tar

# Single-image demo (YOLOv5 detects the person, then mesh recovery + render)
cd demo && python demo.py --gpu 0 --img_path input.png --output_folder output \
    --decoder_setting pytorch --pretrained_model_path ../pretrained_models/osx_l.pth.tar
```

There is no test suite, linter, or build step beyond the `mmpose` install. The README documents the upstream multi-dataset training "settings" (AGORA/UBody/EHF benchmarks); those flags (`--agora_benchmark`, `--ubody_benchmark`) still work and override datasets in `config.py`.

## The `decoder_setting` switch — read this first

`--decoder_setting` (default `normal`) selects **which `Model`/`get_model` implementation is imported**, decided at import time in `common/base.py`:

| setting | module imported | notes |
|---|---|---|
| `normal` | `main/OSX.py` | original OSX; MMCV `mmpose` ViT encoder + MMCV keypoint decoders |
| `wo_face_decoder` | `main/OSX_WoFaceDecoder.py` | drops the face decoder |
| `wo_decoder` | `main/OSX_WoDecoder.py` | encoder-only heads |
| `pytorch` | `main/model_core.py` | **the active rewrite** — pure-PyTorch ViT + DCNv4 heads + custom decoders + frozen backbone |

Train and test **must use the same setting** as the checkpoint, since each path defines different module structures and weight-key layouts. `config.py`'s default is `normal`, so always pass `--decoder_setting` explicitly.

## Architecture / forward pass

All `Model.forward(inputs, targets, meta_info, mode)` implementations share this pipeline (see `main/model_core.py:395` or `main/OSX.py:191`):

1. **Encoder** (ViT-L): image → `img_feat` (B,C,16,12) + `task_tokens`. Tokens are split into `shape / cam / expr / jaw_pose / hand / body_pose` tokens.
2. **Body regressor**: `PositionNet` → body joint heatmaps/coords; `BodyRotationNet` → root+body pose (6D rot) + shape + camera param. `get_camera_trans` turns the cam param into a translation.
3. **BoxNet**: predicts left-hand / right-hand / face bbox centers+sizes from `img_feat` + body heatmaps; `restore_bbox` makes xyxy boxes.
4. **RoI** (`HandRoI`/`FaceRoI`): differentiable feature-level crop + upsample (`cfg.upscale`) into multi-scale feature lists. Left hand is processed flipped, concatenated with right hand.
5. **Hand/face position nets + decoders**: per-part keypoint heatmaps → init coords → **keypoint-guided deformable decoder** refines joint features → regressors output hand pose / jaw pose + expression.
6. **`get_coord`**: runs the SMPL-X layer to get mesh + joints, applies camera projection, and converts to root-relative (wrist-relative for hands, neck-relative for face) coords.
7. `mode=='train'` returns a dict of losses (param, coord, projection, bbox losses — see `model_core.py:464+`); otherwise returns mesh/pose/bbox outputs.

6D rotations are converted via `rot6d_to_axis_angle` (`common/utils/transforms.py`). Joint index sets, parts, and the SMPL-X layer live in `common/utils/human_models.py` (the `smpl_x` singleton).

## The `pytorch` path specifics (`model_core.py`)

This is where the fork diverges from upstream OSX, and the part most likely to need work:

- **Encoder**: `common/nets/vit.py` `StandardViT` (pure PyTorch; 31 task tokens concatenated *after* the patch tokens with **no** pos_embed; `pos_embed` length is `num_patches + 1`, the +1 cls slot broadcast onto every patch) replaces the MMCV `mmpose` ViT used by `OSX.py`. As of 2026-06-27 the port is **bit-identical** to the MMCV ViT — `PatchEmbed` conv must use `padding=2` and `LayerNorm eps=1e-6` to match OSX (a `padding=0` bug had silently made `img_feat` ~24% off; verify with `tool/analysis/encoder_compare.py`).
- **Hand/face heatmap heads**: `DCNv4PositionNet` (`common/nets/DCNv4.py`) replaces `PositionNet` for hand/face (DCNv4-based, `dcnv4_group=4`, `num_blocks=3`).
- **Decoders**: `HandDecoder` / `FaceDecoder` (`common/nets/my_decoder.py`) — deformable-attention decoders with extras like hand-topology attention, face-region attention, and occlusion modules. As of 2026-07-01 `HandDecoder` is a **coordinate refiner**: it regresses a per-layer 2D coordinate with iterative reference-point refinement (logit/sigmoid, mirroring `vit.py` `PoseurDecoder`) and returns `(features, per_layer_coords)` — the coords feed `loss['hand_aux_coord']` (gated by `cfg.hand_aux_coord_loss_weight`). `model_core.py` detaches the decoder's `query_init` by default (`cfg.detach_hand_decoder_query`) to cut the DCNv4 backward-NaN path. `FaceDecoder` still returns features only.
- **Frozen backbone**: `encoder`, `body_position_net`, `body_regressor`, `box_net`, `hand_roi_net`, `face_roi_net` are frozen (`Model.frozen_modules`). Only `hand/face_position_net`, `hand/face_decoder`, and `hand/face_regressor` train. Opt-in flags unfreeze a subset: `--train_hand_roi` moves `hand_roi_net` into the trainable set (route C), `--train_body_shape` re-enables `body_regressor.shape_out`/`cam_out` (T1) — both default off, byte-for-byte no-op when off. `Model.train()` is overridden to keep frozen BN layers in eval mode. `common/base.py` `_verify_freeze_status` asserts this on startup.
- **Two-phase schedule** (`main/train.py`, `--phase1_epochs`, default 10): Phase 1 trains only position nets + decoders; Phase 2 also unfreezes the regressors at a much smaller LR. Each phase rebuilds the optimizer + `CosineAnnealingLR`.

### Checkpoint mechanics (important gotcha)

`Trainer.save_model` writes **lightweight checkpoints containing only trainable-module tensors** (the frozen backbone is omitted to save space). Consequently, loading a `pytorch` checkpoint is always **two steps**:

1. Load `--pretrained_model_path` (original OSX `osx_l.pth.tar`) to fill the frozen backbone, with extensive key remapping — `module.` stripping, `encoder.last_norm.`→`encoder.norm.`, and MMCV-decoder→custom-decoder key mapping (`encoder.pos_embed` is now a direct shape match — `StandardViT.pos_embed` is `(1,193,1024)` like OSX; the old 193→223 expansion is gone).
2. Load `--continue_train_path` (the lightweight snapshot) on top for the trained hand/face modules.

This logic is duplicated across `Trainer._load_pretrained_frozen` / `_load_resume_checkpoint` (`common/base.py`) and `Tester._make_model` / `Demoer._make_model`. `get_model('test1')` (note: not `'test'`) builds the `pytorch` test model. If you change module structure or naming, you will likely need to update these remapping blocks.

## Config conventions (`main/config.py`)

- The `Config` singleton `cfg` is mutated globally. **`cfg.set_args(...)` must be called before `cfg.set_additional_args(...)`**; the latter calls `prepare_dirs`, which creates `output/<exp_name>/{model_dump,vis,log,code,result}` and **copies key source files into `code/` as a per-run backup**.
- `encoder_setting` (`osx_b`/`osx_l`) sets `feat_dim` (768/1024) and the encoder config/pretrained paths.
- **Training datasets are set in `config.py`, not via CLI** (`trainset_3d` / `trainset_2d` near line 27 — currently `['InterHand26M', 'UBody', 'MSCOCO']`). The benchmark flags override these. The test set comes from `--testset` (`EHF` / `UBody` / `InterHand26M` / `HInt` / `HAA500`).
- `cfg.visualization` toggles saving ViT intermediate-layer feature maps / attention during test inference (handled by `common/utils/visualize.py`, output to `vit_vis/`); off by default.

## Datasets

Each dataset has a loader under `data/<NAME>/<NAME>.py` exposing a class named after the directory; `common/base.py` imports them dynamically by name from `cfg.trainset_*`/`cfg.testset`. All annotations follow MSCOCO format. **`HAA500`** (`data/HAA500/HAA500.py`) is a fork addition: a 134-keypoint whole-body dataset read from `dataset/Haa500/`.

## WIP gotchas

- `main/test.py` evaluates the full testset by default; `--max_eval_iters N` caps to N batches for quick checks. `--dump_analysis` / `--dump_encoder_n N` dump per-sample npz consumed by `tool/analysis/` (`bootstrap_ci.py`, `crosspath_compare.py`, `encoder_compare.py`).
- `main/model_core.py` and `main/OSX.py` contain `print` debug statements in the hot forward path.
- `main/OSX.py` (`normal`) runs the **full original OSX path** (MMCV ViT + MMCV decoders + original heads) and is a faithful OSX baseline — it is the source of the 19.58 / 0.219 / 0.487 reference numbers.
- `--debug_vis` in `test.py` renders GT keypoints/bboxes onto images into `output/<exp>/debug_vis/`.
