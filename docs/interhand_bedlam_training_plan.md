# InterHand26M + BEDLAM Training Plan for myosx

> 📌 历史文档（InterHand+BEDLAM 阶段计划）。**早于 2026-06-27 编码器 padding 修复**，且 BEDLAM 已降权。当前项目状态/路线以 `docs/project_overview.md`、`docs/continue.txt` 为准，本文件仅作配方参照。

## 1. Training Goal

Use BEDLAM as full-body SMPL-X rehearsal data and InterHand26M as hand-specific auxiliary data.

The target is:

- improve hand keypoint localization and finger articulation;
- preserve full-body SMPL-X pose, shape, camera, and mesh ability;
- avoid using InterHand annotations as fake full-body SMPL-X supervision.

InterHand26M is MANO-based hand data. BEDLAM is full SMPL-X data. They should be mixed through validity masks, not treated as equivalent full-body labels.

## 1.1 Hand4Whole++ Reference Takeaways

Hand4Whole++ is useful as a design reference, but it should not be copied directly into myosx.

Observed implementation choices in the local `Hand4Whole-plus-plus_RELEASE` reference code:

- dataset mixing is probability-based, not length-based: `InterHand26M / ReInterHand / ARCTIC = 0.4 / 0.3 / 0.3`;
- InterHand samples are explicitly marked as hand-only with flags such as `is_hand_only`, `is_kpt_ih`, and `is_smplx_ih`;
- InterHand disables full-body supervision: `smplx_shape_valid=0`, `smplx_expr_valid=0`;
- MANO `Wrist` is excluded when mapping MANO pose to SMPL-X pose, so InterHand supervises finger joints instead of SMPL-X wrist joints;
- only the hand control module is trainable in the inspected hand-focused setup, while body/face/backbone modules are frozen or eval;
- pose loss is computed after converting axis-angle to rotation matrices, which is more stable than raw axis-angle L1 for articulated fingers;
- it adds InterHand-specific relative hand losses, including wrist-relative and left/right hand relative-vector constraints.

For myosx, the directly useful parts are:

- keep InterHand as hand-only data;
- keep MANO root pose separate from SMPL-X wrist pose;
- use explicit dataset sampling probability;
- add dataset-source flags for logging and conditional losses;
- consider a rotation-matrix hand pose loss after the basic pipeline is stable.

Do not directly import the Hand4Whole++ MANO-to-SMPL-X mesh replacement path unless the project also adopts its WiLoR/MANO hand branch. Our safer route is to use InterHand for hand keypoints and finger pose supervision, while BEDLAM protects full SMPL-X behavior.

## 1.2 Current-Code Review Corrections

The local review in `docs/interhand_bedlam_training_plan_review.md` is mostly correct. It changes the implementation priority of this plan.

Blocking issues in the current repository:

- `cfg.decoder_setting` defaults to `normal`, but `main/train.py` and `common/base.py` currently rely on methods and attributes defined in `main/model_core.py`, such as `freeze_modules`, `training_phase`, and `set_training_phase`. Therefore InterHand+BEDLAM training must use `decoder_setting='pytorch'` unless the same phase/freeze API is ported to `main/OSX.py`.
- `main/train.py` has `PHASE1_EPOCHS = 17`, while the example mixed-finetuning command uses `--end_epoch 10`. With the current phase logic, this keeps `hand_regressor` frozen for the entire run, so InterHand finger-pose supervision cannot update the hand pose head. This must be fixed before any real hand finetuning run.

Items already implemented and no longer first-pass code work:

- InterHand MANO root pose is not used to supervise SMPL-X `L_Wrist/R_Wrist`; the loader already uses `mano_pose[3:48]` for 15+15 finger joints.
- InterHand shape, expression, face bbox, and mesh labels are already masked or dummy-filled.
- The current training loss path already uses valid masks, and there is no direct mesh loss in `mode='train'`.

Additional empirical risk:

- myosx is a top-down whole-body model. InterHand images are hand-centric crops with limited body context. If the frozen `box_net` fails to predict good hand boxes on InterHand images, the hand branch receives badly cropped features and hand losses may become poorly aligned. Stage 0 must therefore visualize model-predicted hand boxes and hand crops on InterHand samples, not only annotation projections.

Additional training-correctness issue:

- InterHand 3D joint coordinates are currently made relative to one shared hand root, while `model_core.get_coord()` makes left-hand finger joints relative to `L_Wrist` and right-hand finger joints relative to `R_Wrist`. For two-hand InterHand samples, the non-root hand receives a shifted 3D cam target. This affects `loss['joint_cam']` and `loss['smplx_joint_cam']`, not 2D projection or MANO finger-pose supervision.
- InterHand wrist joints should not supervise 3D cam coordinates. In model output, `L_Wrist/R_Wrist` remain body/pelvis-relative joints, while InterHand cannot provide body-root-relative wrist positions. Keep wrist 2D/truncation supervision if useful, but set wrist 3D cam valid flags to 0 for InterHand.

## 2. Data Roles

| Dataset | Role | Valid Supervision | Must Not Supervise |
|---|---|---|---|
| BEDLAM | full-body rehearsal and global SMPL-X stabilization | body pose, hand pose, jaw if available, shape, expression if available, SMPL-X joints, SMPL-X mesh target for evaluation/output | none, assuming labels are processed correctly |
| InterHand26M | hand-only local supervision | 42 hand joints, left/right hand bbox, SMPL-X hand finger pose mapped from MANO | SMPL-X shape, expression, face, body pose, full SMPL-X mesh |

InterHand MANO `shape` is not SMPL-X `betas`. InterHand MANO mesh has 778 vertices per hand and cannot be directly used as a 10475-vertex SMPL-X mesh target.

## 3. Required Pre-Training Checks

Run these before any real training.

### 3.1 BEDLAM Label Check

Use a small BEDLAM cap first:

```python
cfg.bedlam_max_samples = 64
cfg.bedlam_skip_missing_images = True
cfg.trainset_3d = ["BEDLAM"]
cfg.trainset_2d = []
```

Confirm:

- images resolve correctly;
- `smplx_pose`, `smplx_shape`, `smplx_expr`, `smplx_joint_cam`, `smplx_mesh_cam` have expected shapes;
- rendered BEDLAM mesh aligns with the image;
- no zero or obviously invalid camera translation.

### 3.2 InterHand Projection and MANO Check

Use the visualization script:

```powershell
python tool\InterHand26M\visualize_interhand26m.py `
  --annot-root E:\InterHand-annos `
  --img-root E:\InterHand-images `
  --num-samples 16 `
  --renderer wireframe
```

If PyTorch3D is available:

```powershell
python tool\InterHand26M\visualize_interhand26m.py `
  --annot-root E:\InterHand-annos `
  --img-root E:\InterHand-images `
  --num-samples 16 `
  --renderer myosx
```

Accept only if:

- 2D keypoints project onto the hands;
- left/right hands are not swapped;
- MANO mesh roughly overlays the same hand as the keypoints;
- interacting-hand samples do not show systematic mirroring;
- wrist-relative depth looks numerically reasonable.

## 4. Dataset Implementation Rules

### 4.1 InterHand Validity Masks

For InterHand samples:

```text
joint_valid:          3D cam valid mask; hand fingers valid, wrists invalid
smplx_joint_valid:    3D cam valid mask; hand fingers valid, wrists invalid
joint_trunc:          2D/truncation mask; mapped visible hand joints may remain valid
smplx_joint_trunc:    2D/truncation mask; mapped visible hand joints may remain valid
smplx_shape_valid:    0
smplx_expr_valid:     0
face_bbox_valid:      0
smplx_mesh_cam:       dummy only, no mesh loss should use it
is_3D:                1
```

Recommended change before training:

- verify that SMPL-X `L_Wrist` and `R_Wrist` pose valid flags remain 0 for InterHand;
- verify that only the 15 left-hand finger joints and 15 right-hand finger joints are pose-valid;
- change InterHand 3D `joint_cam` targets to be per-hand wrist-relative: left hand relative to `L_Wrist`, right hand relative to `R_Wrist`;
- set InterHand `L_Wrist/R_Wrist` valid flags to 0 for 3D cam losses, because InterHand does not provide pelvis/body-relative wrist targets compatible with model output.

Reason: MANO root/global orientation is a hand root orientation, while SMPL-X wrist rotation is part of the full-body kinematic chain. Directly matching them can inject wrong body-chain rotations.

Coordinate reason: `model_core.get_coord()` converts SMPL-X hand finger joints into their own wrist-relative frames before computing `joint_cam` losses. InterHand must use the same convention. A single shared InterHand root is only correct for the hand that owns that root; the other hand is shifted by the inter-wrist vector.

The safer InterHand pose mask is:

```text
valid:   L_Index_1..L_Thumb_3 and R_Index_1..R_Thumb_3
invalid: L_Wrist, R_Wrist, all body joints, jaw
```

### 4.2 BEDLAM Validity Masks

BEDLAM should keep:

```text
smplx_pose_valid:   body + hand + jaw according to labels
smplx_shape_valid:  1
smplx_expr_valid:   1 only if expression labels exist
smplx_joint_valid:  full SMPL-X joint labels
is_3D:              1
```

If BEDLAM processed labels do not contain expression, keep `smplx_expr_valid=0`.

## 5. Data Mixture

Do not directly use all InterHand samples without ratio control. InterHand is much larger and will dominate BEDLAM.

Default conservative mixture:

```text
BEDLAM : InterHand26M = 60 : 40
```

Balanced hand-focused mixture after validation is stable:

```text
BEDLAM : InterHand26M = 50 : 50
```

More aggressive hand finetuning, only if full-body validation remains stable:

```text
BEDLAM : InterHand26M = 40 : 60
```

Current `MultipleDatasets` mixes datasets by length when both are in `trainset_3d`. Therefore use one of these approaches.

### Short-Term Approach: Cap Samples

Set sample caps so lengths approximate the desired ratio:

```python
cfg.trainset_3d = ["BEDLAM", "InterHand26M"]
cfg.trainset_2d = []

cfg.bedlam_max_samples = 120000
cfg.interhand_max_samples = 80000
```

This approximates 60:40 BEDLAM:InterHand.

### Better Approach: Explicit Weighted Sampler

Add a mixed dataset wrapper that samples by probability:

```text
0.6 BEDLAM
0.4 InterHand26M
```

This is preferable once the basic pipeline is validated because it decouples training ratio from dataset size. Hand4Whole++ uses this style: each `__getitem__` first samples a dataset index with `np.random.choice(..., p=db_sample_prob)`, then randomly samples one item from that dataset. For myosx, add a config entry such as:

```python
cfg.trainset_3d_sample_prob = {
    "BEDLAM": 0.6,
    "InterHand26M": 0.4,
}
```

Use the weighted sampler for real runs; use sample caps only for quick smoke tests or temporary experiments.

## 6. Recommended Training Stages

Before running these stages, fix both current-code blockers:

```text
1. Use decoder_setting='pytorch' or port model_core's freeze/phase API to OSX.py.
2. Make hand_regressor trainable during the hand-finetuning phase, or set PHASE1_EPOCHS/end_epoch so Phase 2 is actually reached.
```

### Stage 0: Data Smoke Test

Purpose: catch path, shape, mask, projection, and OOM errors.

Config:

```python
cfg.trainset_3d = ["BEDLAM", "InterHand26M"]
cfg.trainset_2d = []
cfg.bedlam_max_samples = 64
cfg.interhand_max_samples = 64
cfg.bedlam_skip_missing_images = True
cfg.interhand_skip_missing_images = True
```

Run:

```powershell
python main\train.py `
  --decoder_setting pytorch `
  --gpu_ids 0 `
  --lr 1e-5 `
  --end_epoch 1 `
  --train_batch_size 4 `
  --num_thread 2 `
  --exp_name output/debug_interhand_bedlam
```

Pass criteria:

- dataloader starts;
- one epoch finishes;
- no NaN loss;
- InterHand losses only affect valid hand-related entries;
- BEDLAM still produces full SMPL-X targets.
- model-predicted InterHand hand boxes and hand crops are visually plausible.
- InterHand 3D hand finger targets are per-hand wrist-relative;
- InterHand wrist joints are invalid for 3D cam losses.

### Stage 1: BEDLAM Baseline Warmup

Purpose: establish a clean full-body baseline in the current code path.

Use only BEDLAM:

```python
cfg.trainset_3d = ["BEDLAM"]
cfg.trainset_2d = []
cfg.bedlam_max_samples = None
```

Train:

```text
epochs: 1-3
lr:     1e-5 to 3e-5
batch:  as large as stable
```

Update policy:

- if loading a strong pretrained myosx/OSX checkpoint, keep this short;
- train hand/face position nets, decoders, and regressors only if current code freezes the backbone/body branch;
- do not start from random weights.

Save this checkpoint as the full-body rehearsal baseline.

### Stage 2: Hand-Focused Mixed Finetuning

Purpose: improve hands while BEDLAM prevents forgetting.

Data:

```text
BEDLAM : InterHand26M = 60 : 40
```

Suggested config:

```python
cfg.trainset_3d = ["BEDLAM", "InterHand26M"]
cfg.trainset_2d = []
cfg.bedlam_max_samples = 120000
cfg.interhand_max_samples = 80000
cfg.interhand_use_human_annot = True
cfg.interhand_require_mano = True
```

Training length:

```text
epochs: 5-10
```

Learning rates:

```text
hand_position_net:  5e-5 to 1e-4
hand_decoder:       5e-5 to 1e-4
hand_regressor:     5e-6 to 1e-5
face branch:        0 or 1e-6 if BEDLAM expression is valid
body/camera branch: 0 unless doing a low-LR stabilization stage
encoder:            0
```

Important: current `main/train.py` Phase 1 trains position nets and decoders but not regressors. For this stage, include `hand_regressor` with a small LR. Otherwise InterHand pose supervision has limited ability to correct the hand pose head.

Loss behavior:

- InterHand contributes:
  - hand joint 3D loss through valid hand joints;
  - SMPL-X hand finger pose loss;
  - left/right hand bbox loss only if `box_net` is trainable. If `box_net` is frozen, bbox loss is only a diagnostic number and does not improve the model.
- BEDLAM contributes:
  - full SMPL-X pose/shape/joint losses;
  - hand pose as part of full SMPL-X;
  - expression only if valid.

Implementation note from Hand4Whole++: if an explicit weighted sampler is available, prefer that over these caps. The above caps are only an approximation.

Stop or reduce LR if:

- BEDLAM full-body loss increases sharply;
- body/camera output becomes unstable;
- hand left/right swap appears in visual samples.

### Stage 3: Mixed Stabilization

Purpose: consolidate hand gains with stronger full-body rehearsal.

Data:

```text
BEDLAM : InterHand26M = 50 : 50
```

Training length:

```text
epochs: 2-5
```

Learning rates:

```text
hand_position_net:  1e-5 to 3e-5
hand_decoder:       1e-5 to 3e-5
hand_regressor:     1e-6 to 3e-6
face branch:        0 or 1e-6
body/camera branch: 0
encoder:            0
```

If full-body validation is stable, optionally allow very low LR on the last shared layers only. Do not fully unfreeze the encoder.

### Stage 4: Optional Low-LR Global Polish

Only run if Stage 2/3 validation is clean.

Data:

```text
BEDLAM : InterHand26M = 60 : 40
```

Training length:

```text
epochs: 1-2
```

Learning rates:

```text
hand branch:        1e-5
face branch:        1e-6
body/camera branch: 1e-6
encoder last block: 0 to 5e-7
encoder early:      0
```

Skip this stage if there is no full-body validation set.

## 7. Loss Policy

Use existing myosx losses, but enforce correct valid masks.

For InterHand:

```text
L = L_hand_joint_cam
  + L_smplx_hand_joint_cam
  + L_smplx_hand_finger_pose
  + optional L_hand_bbox only when box_net is trainable
```

Recommended InterHand pose policy:

```text
supervise:  L/R hand 15 finger joints each
do not:     supervise SMPL-X L_Wrist/R_Wrist from MANO root pose
```

Recommended InterHand 3D coordinate policy:

```text
left-hand fingers:   relative to L_Wrist
right-hand fingers:  relative to R_Wrist
L_Wrist/R_Wrist:     invalid for 3D cam losses
2D wrist keypoints:  may remain valid through truncation masks if useful
```

Do not include:

```text
L_smplx_shape
L_smplx_expr
L_body_pose
L_face
L_full_smplx_mesh
```

For BEDLAM:

```text
L = L_full_smplx_pose
  + L_smplx_shape
  + L_smplx_joint_cam
  + L_joint_cam
  + L_expr if valid
```

The current model does not appear to apply direct full mesh loss during training, but `smplx_mesh_cam` is still returned for output/evaluation. Keep InterHand mesh dummy values invalid and do not add mesh loss from them.

Optional improvement after the first stable run:

- add a hand-only rotation-matrix pose loss for `L_smplx_hand_finger_pose`;
- keep the existing raw `ParamLoss` for the initial implementation if changing loss code would delay smoke testing;
- if adding the rotation loss, apply it only where `smplx_pose_valid` marks the 30 hand finger joints valid.

Useful but lower-priority InterHand-specific losses:

- hand part-relative 3D joint loss: left hand relative to left wrist, right hand relative to right wrist;
- interacting-hand relative-vector loss: pairwise right-hand to left-hand vectors for samples where both hands are valid.

These losses should be added only after visual checks show that the basic keypoint and pose supervision is correct.

## 8. Evaluation and Monitoring

### 8.1 Always Monitor These Losses Separately

Log by dataset source if possible:

- InterHand hand joint loss;
- InterHand hand pose loss;
- InterHand hand bbox loss as a diagnostic. It only trains the model if `box_net` is included in the optimizer;
- BEDLAM full pose loss;
- BEDLAM shape loss;
- BEDLAM joint loss;
- expression loss if valid.

If the code does not expose dataset source in `meta_info`, add:

```python
meta_info["dataset_id"] = 1 or 0
meta_info["is_interhand"] = 1.0 or 0.0
meta_info["is_bedlam"] = 1.0 or 0.0
meta_info["is_hand_only"] = 1.0 or 0.0
```

This makes debugging mixed training much easier. Add these numeric keys to both BEDLAM and InterHand loaders at the same time; mixed batches require identical dictionary keys and compatible value types.

### 8.2 Visual Checks Every Epoch

Save a fixed panel of samples:

- 8 InterHand samples: keypoints + MANO mesh projection;
- 8 BEDLAM samples: SMPL-X mesh render;
- 8 model predictions after inference: predicted hand boxes, hand crops, and final SMPL-X mesh.

For InterHand, check:

- frozen or trainable `box_net` predicts hand boxes that actually cover the hands;
- the hand crops contain the target hands with enough margin;
- finger articulation improves;
- no systematic left/right swap;
- interacting hands do not collapse into one side;
- hand bbox remains stable.

For BEDLAM, check:

- full-body mesh remains aligned;
- body shape does not drift;
- camera scale remains stable;
- hands do not improve at the cost of body pose collapse.

### 8.3 Validation Gates

Accept Stage 2 only if:

- InterHand hand loss decreases;
- BEDLAM full-body loss does not increase by more than 5-10%;
- qualitative BEDLAM renders remain stable.

Accept final checkpoint only if:

- hand quality improves on InterHand samples;
- body quality remains acceptable on BEDLAM;
- no visible left/right hand convention failure.

## 9. Concrete Config Recommendation

For the first real run after the two blocker fixes:

```python
cfg.decoder_setting = "pytorch"
cfg.trainset_3d = ["BEDLAM", "InterHand26M"]
cfg.trainset_2d = []
cfg.testset = "EHF"

cfg.bedlam_dir = os.environ.get("BEDLAM_DATASET_ROOT", "E:/BEDLAM_Dataset")
cfg.bedlam_img_dir = os.environ.get("BEDLAM_IMG_DIR", cfg.bedlam_dir)
cfg.bedlam_annot_path = os.environ.get("BEDLAM_PROCESSED_LABELS", osp.join(cfg.bedlam_dir, "processed_labels"))
cfg.bedlam_sample_interval = 1
cfg.bedlam_max_samples = 120000
cfg.bedlam_skip_missing_images = True

cfg.interhand_annot_path = os.environ.get("INTERHAND_ANNOS", "E:/InterHand-annos")
cfg.interhand_img_dir = os.environ.get("INTERHAND_IMG_DIR", "E:/InterHand-images")
cfg.interhand_sample_interval = 1
cfg.interhand_max_samples = 80000
cfg.interhand_skip_missing_images = True
cfg.interhand_use_human_annot = True
cfg.interhand_require_mano = True

# Prefer this once the weighted sampler is implemented.
cfg.trainset_3d_sample_prob = {
    "BEDLAM": 0.6,
    "InterHand26M": 0.4,
}
```

Command:

```powershell
python main\train.py `
  --decoder_setting pytorch `
  --gpu_ids 0 `
  --lr 5e-5 `
  --lr_mult 0.1 `
  --end_epoch 10 `
  --train_batch_size 16 `
  --num_thread 4 `
  --pretrained_model_path ..\pretrained_models\osx_l.pth.tar `
  --exp_name output/interhand_bedlam_hand_ft_v1
```

Use smaller batch size if GPU memory is tight.

Do not use the above `--end_epoch 10` with the current unmodified `PHASE1_EPOCHS = 17` if the goal is to train `hand_regressor`. Either modify Phase 1 to include `hand_regressor` at low LR, lower `PHASE1_EPOCHS`, or train for more than 17 epochs and ensure Phase 2 is reached.

## 10. Implementation Adjustments Before Long Training

These are recommended before any multi-day run.

1. Lock the active model entry.

Use `decoder_setting='pytorch'` for the current training script, or port `freeze_modules`, `training_phase`, and `set_training_phase` to `main/OSX.py`.

2. Fix the phase boundary so `hand_regressor` actually trains.

The current `PHASE1_EPOCHS = 17` conflicts with the example `--end_epoch 10`. The preferred fix is to keep `hand_regressor` trainable in the hand-finetuning phase with a small LR and freeze `face_regressor`. Lowering `PHASE1_EPOCHS` or increasing `end_epoch` is acceptable but less explicit.

3. Keep InterHand wrist pose supervision disabled.

This is already implemented in the current InterHand loader. Treat it as a regression check, not as new first-pass code work. InterHand MANO root pose should not supervise SMPL-X `L_Wrist/R_Wrist`; `pose_valid` must remain 0 for wrists.

4. Add numeric dataset-source flags to `meta_info`.

This enables dataset-wise loss logging and makes mixed training debuggable. Prefer numeric flags such as `dataset_id`, `is_interhand`, `is_bedlam`, and `is_hand_only` instead of a string `dataset_name`, because numeric tensors are cleaner with default collate and `DataParallel`.

5. Add explicit weighted sampler or weighted mixed dataset.

This is the highest-priority infrastructure change after the dataloaders are confirmed. Relying on `max_samples` is acceptable for the first smoke run but not ideal for controlled training.

6. Include `hand_regressor` in the hand finetuning optimizer.

Current Phase 1 optimizer only includes `hand_position_net`, `hand_decoder`, `face_position_net`, and `face_decoder`. For InterHand hand pose finetuning, `hand_regressor` should train at low LR.

7. Do not train face branch from InterHand.

Only let BEDLAM expression/jaw labels affect face-related modules, and only if those labels are valid.

8. Visualize InterHand predicted hand boxes and crops.

Because myosx is top-down and InterHand is hand-centric, frozen `box_net` quality on InterHand is a real gate. If predicted boxes are poor, either train `box_net` carefully at low LR with BEDLAM rehearsal or inject InterHand GT hand boxes for the hand ROI path.

9. Consider rotation-matrix hand pose loss.

Hand4Whole++ computes pose loss after converting axis-angle to rotation matrices. This is a good follow-up for myosx hand finetuning, especially if raw axis-angle supervision causes unstable finger rotations.

10. Do not port MANO mesh replacement directly.

Hand4Whole++ combines WiLoR MANO hand meshes with SMPL-X vertices through rigid alignment. myosx does not currently have the same hand mesh branch, so direct porting would add architectural risk. Use InterHand for hand keypoints and finger pose targets instead.

## 11. Failure Modes and Fixes

| Symptom | Likely Cause | Fix |
|---|---|---|
| Hand improves but body collapses | InterHand ratio too high or body/shared modules unfrozen | increase BEDLAM to 50-60%, freeze body/encoder |
| Left and right hands swapped | InterHand flip/mapping convention issue | inspect fixed visual samples, check `flip_pairs` and MANO side mapping |
| Fingers curl unnaturally | MANO mean pose or joint order mismatch | verify `use_pca=False`, `flat_hand_mean=False`, and 15-joint order |
| Loss becomes NaN | bad camera/depth, missing image, invalid bbox | run capped smoke test, skip bad samples |
| Full-body shape drifts | InterHand shape dummy accidentally valid | ensure `smplx_shape_valid=0` for InterHand |
| Face expression degrades | face branch trained without valid face labels | freeze face branch or train only on BEDLAM valid expr |
| Hand pose loss decreases but rendered fingers look unstable | raw axis-angle loss or MANO/SMPL-X joint order issue | verify 30 finger joint mapping, then consider rotation-matrix hand pose loss |
| Training crashes before first batch | `decoder_setting='normal'` loads `OSX.py`, but train/base expects `model_core.py` phase/freeze API | use `--decoder_setting pytorch` or port the API to `OSX.py` |
| Hand pose head never improves | `PHASE1_EPOCHS=17` and `end_epoch<=17` keep `hand_regressor` frozen | train `hand_regressor` in Phase 1 at low LR, lower `PHASE1_EPOCHS`, or run into Phase 2 |
| Hand losses decrease but crops are wrong | frozen top-down `box_net` does not localize InterHand hand-centric images | visualize predicted boxes/crops; train `box_net` cautiously or inject GT hand boxes |
| InterHand 3D hand loss gives biased gradients on two-hand samples | InterHand targets use one shared wrist root but model uses per-hand wrist-relative hand coordinates | make InterHand `joint_cam` per-hand wrist-relative and disable wrist joints for 3D cam loss |

## 12. Recommended Final Recipe

Use this sequence:

```text
Stage 0:
  64 BEDLAM + 64 InterHand, 1 epoch smoke test.
  Must use decoder_setting=pytorch or a ported OSX.py phase API.
  Must visualize predicted InterHand hand boxes and crops.
  Must verify InterHand per-hand wrist-relative 3D targets.

Stage 1:
  BEDLAM only, 1-3 epochs, low LR, establish full-body baseline.

Stage 2:
  60% BEDLAM + 40% InterHand, 5-10 epochs.
  Train hand position net, hand decoder, hand regressor.
  Keep encoder/body frozen.

Stage 3:
  50% BEDLAM + 50% InterHand, 2-5 epochs.
  Lower LR, stabilize body and camera.

Stage 4:
  Optional 60% BEDLAM + 40% InterHand, 1-2 epochs.
  Very low LR global polish only if validation is stable.
```

Main rule:

```text
InterHand teaches hands.
BEDLAM protects full SMPL-X.
Validity masks decide what each sample is allowed to supervise.
```

## 13. Code Modification Plan

This section translates the training plan into concrete file-level changes. The first implementation pass should focus on controllable mixed training; hand-specific loss upgrades should come after the dataloaders, masks, and optimizer behavior are verified.

### 13.1 Active Model Entry

Files:

- `main/config.py`
- `common/base.py`
- `main/OSX.py`
- `main/model_core.py`

Blocking issue:

- `cfg.decoder_setting = 'normal'` makes `common/base.py` load `main/OSX.py`;
- the current two-phase freeze/unfreeze logic in `main/train.py` and `common/base.py` requires the API implemented in `main/model_core.py`;
- if code changes are made in the inactive model file, training will not use them.

Recommended decision:

```python
cfg.decoder_setting = 'pytorch'
```

Then use `main/model_core.py` as the active training model for InterHand+BEDLAM. If `normal` mode must remain active, synchronize the relevant phase/freeze/optimizer changes back into `main/OSX.py` instead.

This is not optional for the current `main/train.py`: with `decoder_setting='normal'`, `OSX.py` lacks `freeze_modules`, `training_phase`, and `set_training_phase`, so training can fail before the first batch.

### 13.2 Training Config

File:

- `main/config.py`

Add or update:

```python
decoder_setting = 'pytorch'
trainset_3d = ['BEDLAM', 'InterHand26M']
trainset_2d = []

use_weighted_dataset_sampling = True
trainset_3d_sample_prob = {
    'BEDLAM': 0.6,
    'InterHand26M': 0.4,
}

bedlam_max_samples = 120000
interhand_max_samples = 80000
interhand_skip_missing_images = True

use_hand_rotmat_pose_loss = False
hand_rotmat_pose_loss_weight = 1.0
```

Keep `use_hand_rotmat_pose_loss=False` for the first smoke test. The first goal is to verify that mixed data, valid masks, optimizer groups, and InterHand predicted hand crops are correct.

### 13.3 Weighted Dataset Sampling

Files:

- `data/dataset.py`
- `common/base.py`

In `data/dataset.py`, extend `MultipleDatasets`:

```python
class MultipleDatasets(Dataset):
    def __init__(self, dbs, make_same_len=True, sample_prob=None):
        self.dbs = dbs
        self.db_num = len(self.dbs)
        self.sample_prob = sample_prob
        ...
```

When `sample_prob is not None`, `__getitem__` should:

```python
db_idx = int(np.random.choice(np.arange(self.db_num), p=self.sample_prob))
data_idx = random.randint(0, len(self.dbs[db_idx]) - 1)
return self.dbs[db_idx][data_idx]
```

`__len__` can initially return `sum(len(db) for db in self.dbs)`. A separate fixed `itr_per_epoch` config can be added later if epoch length needs to be decoupled from dataset size.

In `common/base.py::_make_batch_generator`, build the probability list from `cfg.trainset_3d_sample_prob`:

```python
sample_prob = [cfg.trainset_3d_sample_prob[name] for name in cfg.trainset_3d]
sample_prob = np.asarray(sample_prob, dtype=np.float32)
sample_prob = sample_prob / sample_prob.sum()
trainset3d_loader = [MultipleDatasets(trainset3d_loader, make_same_len=False, sample_prob=sample_prob)]
```

Only enable this path when multiple 3D datasets are present and `cfg.use_weighted_dataset_sampling=True`. Keep 2D datasets out of this first implementation pass.

### 13.4 Dataset Meta Flags

Files:

- `data/InterHand26M/InterHand26M.py`
- `data/BEDLAM/BEDLAM.py`

For InterHand samples, add numeric source flags:

```python
meta_info.update({
    'dataset_id': 1,
    'is_interhand': float(True),
    'is_bedlam': float(False),
    'is_hand_only': float(True),
})
```

Also verify these existing masks:

```text
smplx_shape_valid = 0
smplx_expr_valid  = 0
face_bbox_valid   = 0
L_Wrist/R_Wrist pose valid = 0
only 15+15 hand finger joints have pose valid = 1
```

These masks are already correct in the current loader. Keep them as regression checks and avoid rewriting the pose mapping unless a check fails.

Required coordinate fix:

- replace the current shared-root InterHand 3D normalization with per-hand normalization;
- right-hand joints should subtract `R_Wrist`;
- left-hand joints should subtract `L_Wrist`;
- set InterHand `L_Wrist/R_Wrist` 3D cam valid flags to 0;
- if a hand wrist is invalid, mark that hand's 3D cam valid flags invalid instead of borrowing the other hand's root;
- keep 2D truncation masks separate from 3D cam valid masks so useful 2D wrist supervision is not accidentally removed.

The current shared-root convention is not compatible with `model_core.get_coord()` for two-hand InterHand samples.

For BEDLAM samples, add:

```python
meta_info.update({
    'dataset_id': 0,
    'is_interhand': float(False),
    'is_bedlam': float(True),
    'is_hand_only': float(False),
})
```

BEDLAM should keep full-body valid supervision according to the processed labels.

### 13.5 Hand-Finetuning Optimizer

Files:

- `main/train.py`
- `main/model_core.py`

Blocking issue:

- Phase 1 trains `hand_position_net`, `hand_decoder`, `face_position_net`, and `face_decoder`;
- InterHand finger pose supervision needs `hand_regressor` to receive gradients;
- face modules should not be trained from InterHand.
- `PHASE1_EPOCHS = 17`; with `--end_epoch 10`, Phase 2 is never reached, so the current `hand_regressor` remains frozen for the whole run.

Recommended Phase 1 for InterHand+BEDLAM:

```text
train:
  hand_position_net
  hand_decoder
  hand_regressor with low LR

freeze:
  encoder
  body_position_net
  body_regressor
  box_net initially
  face_position_net
  face_decoder
  face_regressor
```

Suggested optimizer groups:

```python
torch.optim.Adam([
    {
        'params': list(model.hand_position_net.parameters()) +
                  list(model.hand_decoder.parameters()),
        'lr': cfg.lr,
    },
    {
        'params': list(model.hand_regressor.parameters()),
        'lr': cfg.lr * cfg.lr_mult,
    },
])
```

Phase 2 can optionally allow very low learning rate on body/camera modules, but only after visual and numeric validation show that Stage 1/2 is stable.

Do not rely on `--end_epoch 10` to train `hand_regressor` unless Phase 1 has been modified to include it. Otherwise the hand pose head receives no parameter updates.

### 13.6 Loss Routing

Files:

- `main/model_core.py`
- `common/nets/loss.py`

Required behavior:

- InterHand must contribute only valid hand keypoint, hand bbox, and hand finger pose losses;
- InterHand must not contribute shape, expression, face, body pose, or full mesh losses;
- BEDLAM must keep full-body SMPL-X losses.

The first implementation should rely on valid masks and avoid adding dataset-specific branches unless a loss cannot be masked cleanly.

Current status:

- InterHand shape/expression/face masks are already 0;
- InterHand dummy mesh is not used by a training mesh loss;
- InterHand wrist pose supervision is already disabled.

Therefore loss-routing work in the first pass should focus on verification and logging, not rewriting existing valid-mask logic.

Optional second-pass improvement:

```python
class RotMatParamLoss(nn.Module):
    ...
```

Implement it in `common/nets/loss.py` using `common.utils.geometry.batch_rodrigues`, then apply it only to the 30 SMPL-X hand finger joints where `smplx_pose_valid` is 1. Do not replace the entire `ParamLoss` globally in the first pass.

### 13.7 Batch Check Script

Optional but recommended file:

- `tool/InterHand26M/check_interhand_bedlam_batch.py`

The script should:

- construct the same training dataloader as `main/train.py`;
- sample several batches;
- print observed `is_interhand/is_bedlam` ratio;
- check InterHand `L_Wrist/R_Wrist` pose valid is 0;
- check InterHand `L_Wrist/R_Wrist` 3D cam valid is 0;
- check InterHand left-hand finger `joint_cam` is relative to `L_Wrist` and right-hand finger `joint_cam` is relative to `R_Wrist`;
- check InterHand shape/expression/face valid flags are 0;
- check BEDLAM shape valid is 1;
- report tensor shapes for `smplx_pose`, `smplx_joint_cam`, `joint_cam`, hand bbox targets, and valid masks.
- optionally run one model forward pass in test mode and save predicted InterHand hand boxes/crops.

This script should run before any long training job.

### 13.8 Implementation Order

Recommended order:

1. Lock the active model entry: `decoder_setting='pytorch'` or port the phase/freeze API to `OSX.py`.
2. Fix the phase boundary so `hand_regressor` trains during hand finetuning.
3. Add config keys in `main/config.py`.
4. Fix InterHand 3D cam targets to be per-hand wrist-relative and disable InterHand wrist 3D cam supervision.
5. Add numeric dataset flags in both `InterHand26M.py` and `BEDLAM.py`, preserving identical dict keys.
6. Add the batch check script and verify masks plus InterHand 3D coordinate convention.
7. Run 64 BEDLAM + 64 InterHand smoke test with `--decoder_setting pytorch`.
8. Visualize predicted InterHand hand boxes/crops; continue only if they are plausible.
9. Implement weighted sampling in `data/dataset.py`.
10. Wire sampling probabilities in `common/base.py`.
11. Add rotation-matrix hand pose loss only after the basic mixed training path is stable.

Minimum first-pass files:

```text
main/config.py
data/dataset.py
common/base.py
data/InterHand26M/InterHand26M.py
data/BEDLAM/BEDLAM.py
main/train.py
main/model_core.py
tool/InterHand26M/check_interhand_bedlam_batch.py
```

Second-pass files:

```text
common/nets/loss.py
```
