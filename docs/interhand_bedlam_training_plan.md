# InterHand26M + BEDLAM Training Plan for myosx

## 1. Training Goal

Use BEDLAM as full-body SMPL-X rehearsal data and InterHand26M as hand-specific auxiliary data.

The target is:

- improve hand keypoint localization and finger articulation;
- preserve full-body SMPL-X pose, shape, camera, and mesh ability;
- avoid using InterHand annotations as fake full-body SMPL-X supervision.

InterHand26M is MANO-based hand data. BEDLAM is full SMPL-X data. They should be mixed through validity masks, not treated as equivalent full-body labels.

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
joint_valid:          only mapped hand joints valid
smplx_joint_valid:    only mapped hand joints valid
smplx_shape_valid:    0
smplx_expr_valid:     0
face_bbox_valid:      0
smplx_mesh_cam:       dummy only, no mesh loss should use it
is_3D:                1
```

Recommended change before training:

- do not supervise SMPL-X `L_Wrist` and `R_Wrist` pose from MANO root pose by default;
- supervise only the 15 left-hand finger joints and 15 right-hand finger joints.

Reason: MANO root/global orientation is a hand root orientation, while SMPL-X wrist rotation is part of the full-body kinematic chain. Directly matching them can inject wrong body-chain rotations.

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

Target mixture:

```text
InterHand26M : BEDLAM = 60 : 40
```

More conservative if body quality drops:

```text
InterHand26M : BEDLAM = 50 : 50
```

More aggressive hand finetuning:

```text
InterHand26M : BEDLAM = 70 : 30
```

Current `MultipleDatasets` mixes datasets by length when both are in `trainset_3d`. Therefore use one of these approaches.

### Short-Term Approach: Cap Samples

Set sample caps so lengths approximate the desired ratio:

```python
cfg.trainset_3d = ["BEDLAM", "InterHand26M"]
cfg.trainset_2d = []

cfg.bedlam_max_samples = 80000
cfg.interhand_max_samples = 120000
```

This approximates 60:40 InterHand:BEDLAM.

### Better Approach: Explicit Weighted Sampler

Add a mixed dataset wrapper that samples by probability:

```text
0.6 InterHand26M
0.4 BEDLAM
```

This is preferable once the basic pipeline is validated because it decouples training ratio from dataset size.

## 6. Recommended Training Stages

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
InterHand26M : BEDLAM = 60 : 40
```

Suggested config:

```python
cfg.trainset_3d = ["BEDLAM", "InterHand26M"]
cfg.trainset_2d = []
cfg.bedlam_max_samples = 80000
cfg.interhand_max_samples = 120000
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
  - left/right hand bbox loss if bbox is valid.
- BEDLAM contributes:
  - full SMPL-X pose/shape/joint losses;
  - hand pose as part of full SMPL-X;
  - expression only if valid.

Stop or reduce LR if:

- BEDLAM full-body loss increases sharply;
- body/camera output becomes unstable;
- hand left/right swap appears in visual samples.

### Stage 3: Mixed Stabilization

Purpose: consolidate hand gains with stronger full-body rehearsal.

Data:

```text
InterHand26M : BEDLAM = 50 : 50
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
InterHand26M : BEDLAM = 40 : 60
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
  + L_hand_bbox
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

## 8. Evaluation and Monitoring

### 8.1 Always Monitor These Losses Separately

Log by dataset source if possible:

- InterHand hand joint loss;
- InterHand hand pose loss;
- InterHand hand bbox loss;
- BEDLAM full pose loss;
- BEDLAM shape loss;
- BEDLAM joint loss;
- expression loss if valid.

If the code does not expose dataset source in `meta_info`, add:

```python
meta_info["dataset_name"] = "InterHand26M" or "BEDLAM"
```

This makes debugging mixed training much easier.

### 8.2 Visual Checks Every Epoch

Save a fixed panel of samples:

- 8 InterHand samples: keypoints + MANO mesh projection;
- 8 BEDLAM samples: SMPL-X mesh render;
- 8 model predictions after inference: hand crop and final SMPL-X mesh.

For InterHand, check:

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

For the first real run:

```python
cfg.trainset_3d = ["BEDLAM", "InterHand26M"]
cfg.trainset_2d = []
cfg.testset = "EHF"

cfg.bedlam_dir = os.environ.get("BEDLAM_DATASET_ROOT", "/workspace/BEDLAM_Dataset")
cfg.bedlam_img_dir = os.environ.get("BEDLAM_IMG_DIR", cfg.bedlam_dir)
cfg.bedlam_annot_path = os.environ.get("BEDLAM_PROCESSED_LABELS", osp.join(cfg.bedlam_dir, "processed_labels"))
cfg.bedlam_sample_interval = 1
cfg.bedlam_max_samples = 80000
cfg.bedlam_skip_missing_images = True

cfg.interhand_annot_path = os.environ.get("INTERHAND_ANNOS", "E:/InterHand-annos")
cfg.interhand_img_dir = os.environ.get("INTERHAND_IMG_DIR", "E:/InterHand-images")
cfg.interhand_sample_interval = 1
cfg.interhand_max_samples = 120000
cfg.interhand_skip_missing_images = True
cfg.interhand_use_human_annot = True
cfg.interhand_require_mano = True
```

Command:

```powershell
python main\train.py `
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

## 10. Implementation Adjustments Before Long Training

These are recommended before any multi-day run.

1. Disable InterHand wrist pose supervision.

Current InterHand loader maps MANO root pose to SMPL-X `L_Wrist/R_Wrist`. Change it so wrist pose target is filled if useful for debugging, but `pose_valid` remains 0 for wrists.

2. Add dataset name to `meta_info`.

This enables dataset-wise loss logging and makes mixed training debuggable.

3. Add explicit weighted sampler or weighted mixed dataset.

Relying on `max_samples` is acceptable for the first run but not ideal.

4. Include `hand_regressor` in the hand finetuning optimizer.

Current Phase 1 optimizer only includes `hand_position_net`, `hand_decoder`, `face_position_net`, and `face_decoder`. For InterHand hand pose finetuning, `hand_regressor` should train at low LR.

5. Do not train face branch from InterHand.

Only let BEDLAM expression/jaw labels affect face-related modules, and only if those labels are valid.

## 11. Failure Modes and Fixes

| Symptom | Likely Cause | Fix |
|---|---|---|
| Hand improves but body collapses | InterHand ratio too high or body/shared modules unfrozen | increase BEDLAM to 50-60%, freeze body/encoder |
| Left and right hands swapped | InterHand flip/mapping convention issue | inspect fixed visual samples, check `flip_pairs` and MANO side mapping |
| Fingers curl unnaturally | MANO mean pose or joint order mismatch | verify `use_pca=False`, `flat_hand_mean=False`, and 15-joint order |
| Loss becomes NaN | bad camera/depth, missing image, invalid bbox | run capped smoke test, skip bad samples |
| Full-body shape drifts | InterHand shape dummy accidentally valid | ensure `smplx_shape_valid=0` for InterHand |
| Face expression degrades | face branch trained without valid face labels | freeze face branch or train only on BEDLAM valid expr |

## 12. Recommended Final Recipe

Use this sequence:

```text
Stage 0:
  64 BEDLAM + 64 InterHand, 1 epoch smoke test.

Stage 1:
  BEDLAM only, 1-3 epochs, low LR, establish full-body baseline.

Stage 2:
  60% InterHand + 40% BEDLAM, 5-10 epochs.
  Train hand position net, hand decoder, hand regressor.
  Keep encoder/body frozen.

Stage 3:
  50% InterHand + 50% BEDLAM, 2-5 epochs.
  Lower LR, stabilize body and camera.

Stage 4:
  Optional 40% InterHand + 60% BEDLAM, 1-2 epochs.
  Very low LR global polish only if validation is stable.
```

Main rule:

```text
InterHand teaches hands.
BEDLAM protects full SMPL-X.
Validity masks decide what each sample is allowed to supervise.
```
