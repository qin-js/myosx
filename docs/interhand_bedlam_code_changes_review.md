# InterHand26M + BEDLAM Code Changes Review

This document summarizes the implementation changes made for InterHand26M + BEDLAM finetuning and the later BEDLAM SMPL-X `flat_hand_mean` compatibility fix.

## 1. `main/config.py`

Purpose:

- Make InterHand26M + BEDLAM the default mixed finetuning setup.
- Use Linux-local dataset paths instead of the Windows paths shown in the plan.
- Force the active training model to `model_core.py` by default.
- Add configuration switches for weighted sampling, hand-regressor finetuning, and BEDLAM hand-pose compatibility.

Changes:

- Added `InterHand26M` to `dataset_list`.
- Changed default training datasets:

```python
trainset_3d = ['BEDLAM', 'InterHand26M']
trainset_2d = []
testset = 'EHF'
```

- Updated BEDLAM defaults to local Linux paths with environment-variable overrides:

```python
bedlam_dir = os.environ.get('BEDLAM_DATASET_ROOT', '/workspace/BEDLAM_Dataset')
bedlam_img_dir = os.environ.get('BEDLAM_IMG_DIR', bedlam_dir)
bedlam_annot_path = os.environ.get('BEDLAM_PROCESSED_LABELS', osp.join(bedlam_dir, 'processed_labels'))
```

- Set BEDLAM defaults for mixed training:

```python
bedlam_max_samples = 120000
bedlam_skip_missing_images = True
bedlam_convert_flat_hand_mean = True
```

- Updated InterHand defaults to local Linux paths:

```python
interhand_annot_path = os.environ.get('INTERHAND_ANNOS', '/workspace/OpenDataLab___InterHand2_dot_6M')
interhand_img_dir = os.environ.get('INTERHAND_IMG_DIR', '/workspace/OpenDataLab___InterHand2_dot_6M/raw/InterHand2.6M_5fps_batch1/images')
interhand_max_samples = 80000
interhand_skip_missing_images = True
```

- Added explicit weighted 3D dataset sampling config:

```python
use_weighted_dataset_sampling = True
trainset_3d_sample_prob = {
    'BEDLAM': 0.6,
    'InterHand26M': 0.4,
}
```

- Added hand-finetuning controls:

```python
phase1_epochs = 10
phase1_train_hand_regressor = True
train_face_modules = False
```

- Rotation-matrix hand pose loss flags — now IMPLEMENTED, default OFF (see §12.1
  for the implementation; the flag values themselves are unchanged):

```python
use_hand_rotmat_pose_loss = False
hand_rotmat_pose_loss_weight = 1.0
```

- Changed default decoder setting:

```python
decoder_setting = 'pytorch'
```

Reason:

- The current training code depends on `model_core.py` APIs such as `freeze_modules`, `training_phase`, and `set_training_phase`; `decoder_setting='normal'` would load `OSX.py` and can fail.

- Updated experiment code backup list to include files changed by this work:

```python
main/model_core.py
data/BEDLAM/BEDLAM.py
data/InterHand26M/InterHand26M.py
```

## 2. `data/dataset.py`

Purpose:

- Add explicit probability-based dataset sampling.
- Decouple BEDLAM/InterHand training ratio from dataset length.

Changes:

- Extended `MultipleDatasets` constructor:

```python
def __init__(self, dbs, make_same_len=True, sample_prob=None):
```

- Added validation and normalization for `sample_prob`.
- Added weighted sampling behavior in `__getitem__`:

```python
db_idx = int(np.random.choice(np.arange(self.db_num), p=self.sample_prob))
data_idx = random.randint(0, len(self.dbs[db_idx]) - 1)
return self.dbs[db_idx][data_idx]
```

Behavior:

- If `sample_prob` is not provided, existing `make_same_len` and length-based behavior is preserved.
- `__len__` is unchanged; for `make_same_len=False`, it still returns the sum of child dataset lengths.

## 3. `common/base.py`

Purpose:

- Wire `cfg.trainset_3d_sample_prob` into the training DataLoader construction.
- Keep 2D dataset behavior unchanged.
- Make `num_thread=0` compatible with PyTorch DataLoader options.

Changes:

- Imported `numpy`.
- Added helper:

```python
def _get_dataset_sample_prob(self, dataset_names, prob_cfg):
```

This helper:

- verifies every dataset has a configured probability;
- rejects negative probabilities;
- normalizes probabilities to sum to 1.

- In `_make_batch_generator`, when multiple 3D datasets are present and `cfg.use_weighted_dataset_sampling=True`, constructs:

```python
MultipleDatasets(trainset3d_loader, make_same_len=False, sample_prob=sample_prob)
```

- Logs the active weighted sampling ratio, e.g.:

```text
Using weighted 3D dataset sampling: BEDLAM=0.600, InterHand26M=0.400
```

- Changed DataLoader worker options:

```python
persistent_workers=cfg.num_thread > 0
prefetch_factor=3 if cfg.num_thread > 0 else None
```

Reason:

- `persistent_workers=True` and `prefetch_factor` are invalid or problematic when `num_workers=0`.

## 4. `data/InterHand26M/InterHand26M.py`

Purpose:

- Keep InterHand as hand-only supervision.
- Fix the 3D coordinate convention mismatch between InterHand targets and `model_core.get_coord()`.
- Add numeric dataset-source flags for mixed-batch logging/debugging.

Changes:

- Imported `transform_joint_to_other_db`.

- Replaced the old shared-root hand coordinate normalization:

```python
_root_relative_interhand(...)
```

with per-hand wrist-relative normalization:

```python
_per_hand_relative_interhand(...)
```

Behavior:

- Right-hand joints subtract `R_Wrist`.
- Left-hand joints subtract `L_Wrist`.
- If a hand wrist is invalid, that hand's 3D coordinate valid flags are set to invalid instead of borrowing the other wrist.
- `R_Wrist` and `L_Wrist` 3D cam valid flags are set to 0 after relative conversion.

Reason:

- `model_core.get_coord()` produces hand finger 3D targets relative to each hand's own wrist.
- InterHand previously used one shared hand root; this shifts the non-root hand in two-hand samples and gives biased 3D cam loss.

- Added `_process_coord_valid(...)` to transform 3D coordinate valid masks through flip and InterHand-to-SMPL-X joint-name mapping without reprocessing coordinates.

- Preserved 2D/truncation supervision separately:

```python
smplx_joint_trunc = joint_trunc.copy()
```

Reason:

- Wrist 3D cam supervision should be disabled for InterHand, but visible 2D wrist supervision can still be useful.

- Existing InterHand SMPL-X parameter masks remain:

```python
smplx_shape_valid = 0
smplx_expr_valid = 0
face_bbox_valid = 0
```

- Existing MANO-to-SMPL-X finger-pose mapping remains:

```python
pose[smpl_x.orig_joint_part['lhand']] = mano_pose[3:48]
pose[smpl_x.orig_joint_part['rhand']] = mano_pose[3:48]
```

This excludes MANO wrist/global orientation from SMPL-X wrist supervision.

- Added numeric mixed-dataset flags to `meta_info`:

```python
dataset_id = 1
is_interhand = True
is_bedlam = False
is_hand_only = True
```

## 5. `data/BEDLAM/BEDLAM.py`

Purpose:

- Keep BEDLAM as full-body SMPL-X rehearsal data.
- Add mixed-dataset flags.
- Fix BEDLAM hand pose compatibility with myosx SMPL-X defaults.

### 5.1 Dataset Flags

Added numeric mixed-dataset flags to `meta_info`:

```python
dataset_id = 0
is_interhand = False
is_bedlam = True
is_hand_only = False
```

### 5.2 `flat_hand_mean` Compatibility Fix

Problem:

- myosx creates global SMPL-X layers in `common/utils/human_models.py` without explicitly setting `flat_hand_mean`, so the default is `flat_hand_mean=False`.
- BEDLAM pose annotations are compatible with `flat_hand_mean=True`.
- Feeding BEDLAM hand poses directly into myosx's default layer makes the hand pose offset wrong.

Implemented fix:

- Added `cfg.bedlam_convert_flat_hand_mean`, default `True`.
- During BEDLAM loader initialization, extract myosx hand means once:

```python
pose_mean = smpl_x.layer["neutral"].pose_mean.detach().cpu().numpy().reshape(-1)
lhand_mean = pose_mean[75:120]
rhand_mean = pose_mean[120:165]
```

- Added `_convert_smplx_param_flat_hand_mean(...)`.

It converts BEDLAM hand poses from `flat_hand_mean=True` convention to myosx default convention:

```python
lhand_pose = lhand_pose - lhand_mean
rhand_pose = rhand_pose - rhand_mean
```

Reason:

- The SMPL-X layer with `flat_hand_mean=False` adds `pose_mean` internally.
- Subtracting the hand mean before passing BEDLAM hand pose into myosx makes the final internal full pose match the BEDLAM `flat_hand_mean=True` interpretation.

Double-conversion guard:

```python
flat_hand_mean_converted = True
```

This prevents repeatedly subtracting the mean if a record is converted during load and later passed through `__getitem__`.

Coverage:

- `.npz` path: conversion happens when `smplx_param` is created.
- `.pkl` cache path: conversion happens after loading each cached record.
- `__getitem__`: conversion is called again as a safety net; the marker prevents duplicate conversion.

## 6. `main/model_core.py`

Purpose:

- Align model phase switching with InterHand hand-pose finetuning.

Changes:

- Updated `set_training_phase(...)`.

Phase 1:

- Hand-focused finetuning.
- `hand_regressor` is trainable when `cfg.phase1_train_hand_regressor=True`.
- `face_regressor` is trainable only when `cfg.train_face_modules=True`.

Phase 2:

- `hand_regressor` remains trainable.
- `face_regressor` remains controlled by `cfg.train_face_modules`.

Reason:

- The old Phase 1 froze regressors. With `--end_epoch 10` and `PHASE1_EPOCHS=17`, `hand_regressor` would never train, so InterHand finger-pose supervision could not update the hand pose head.

## 7. `main/train.py`

Purpose:

- Make `pytorch` decoder the CLI default.
- Make Phase 1 actually train the hand pose head.
- Add CLI switches for smoke tests and ablations.
- Improve mixed-training logging.

Changes:

- Replaced fixed constants:

```python
PHASE1_EPOCHS = 17
PHASE2_EPOCHS = 50
```

with:

```python
DEFAULT_PHASE1_EPOCHS = 10
```

- Changed CLI default:

```python
--decoder_setting pytorch
```

- Added CLI arguments:

```text
--phase1_epochs
--train_face_modules
--no_phase1_hand_regressor
--bedlam_max_samples
--interhand_max_samples
--disable_weighted_dataset_sampling
```

- Added hard guard:

```python
if cfg.decoder_setting != 'pytorch':
    raise ValueError(...)
```

Reason:

- Current training code relies on `model_core.py` phase/freeze API.

- Added helper functions:

```python
_set_module_trainable(...)
_collect_module_params(...)
_configure_training_phase(...)
```

- Phase 1 optimizer now trains:

```text
hand_position_net
hand_decoder
hand_regressor at cfg.lr * cfg.lr_mult
```

- Face modules are excluded unless `--train_face_modules` is explicitly set.

- Phase 2 keeps the same configured modules but lowers learning rates.

- Gradient clipping now only uses parameters present in the current optimizer.

- Training logs now include batch composition when flags are present:

```text
batch_interhand: N
batch_bedlam: M
```

## 8. Validation Performed

Commands run:

```bash
python -m compileall main/config.py main/train.py common/base.py data/dataset.py data/InterHand26M/InterHand26M.py data/BEDLAM/BEDLAM.py main/model_core.py
```

Result:

- Passed.

Command run:

```bash
cd /workspace/myosx/main
python train.py --help
```

Result:

- Passed.
- New CLI arguments are registered.

Weighted sampler lightweight check:

- Built a dummy `MultipleDatasets` with two small fake datasets and `sample_prob=[0.6, 0.4]`.
- Confirmed both datasets are sampled and `__len__` remains length-sum based.

BEDLAM flat-hand conversion lightweight check:

- Called `_convert_smplx_param_flat_hand_mean(...)` with synthetic left/right hand poses and synthetic means.
- Confirmed left and right hand poses subtract the correct means.
- Confirmed repeated calls do not subtract again because `flat_hand_mean_converted=True`.

Known validation limitation:

- A full BEDLAM + InterHand loader smoke test was attempted with tiny caps, but the process was killed with exit code `137`, likely due to memory pressure during large annotation and SMPL-X initialization.
- The code therefore has compile-level and targeted unit-level validation, but a full first-batch GPU run still needs to be performed in an environment with enough memory.

## 9. Post-Review Fixes (Code-Review Follow-up)

These two changes were added after the code review of the InterHand+BEDLAM work,
to close gaps flagged in `docs/interhand_bedlam_training_plan.md` (§13.7 batch
check script, and the per-worker RNG correctness for weighted sampling).

### 9.1 Per-Worker numpy/random Seeding (`common/base.py`)

Problem:

- `MultipleDatasets.__getitem__` uses `np.random.choice` to pick a dataset under
  weighted sampling.
- PyTorch seeds python `random` and `torch` per DataLoader worker, but **not**
  numpy. With `num_thread>0`, every worker would otherwise share the same numpy
  RNG state and draw correlated dataset-selection sequences.

Fix:

- Added a module-level `_seed_worker(worker_id)` that derives a per-worker seed
  from `torch.initial_seed()` and seeds both `np.random` and `random`:

```python
def _seed_worker(worker_id):
    worker_seed = torch.initial_seed() % (2 ** 32)
    np.random.seed(worker_seed)
    random.seed(worker_seed)
```

- Wired into the training `DataLoader` in `_make_batch_generator`:

```python
worker_init_fn=_seed_worker if cfg.num_thread > 0 else None,
```

- Added `import random` / `import torch` at the top of `common/base.py`
  (`torch` was previously only bound indirectly via `import torch.optim`).

Effect:

- The actual sampled item index already used per-worker-seeded `random.randint`,
  so net data diversity was preserved; this fix decorrelates the *dataset
  selection* pattern across workers so the realized BEDLAM:InterHand ratio is
  clean and reproducible.

### 9.2 Stage-0 Batch / Mask Checker (`tool/InterHand26M/check_interhand_bedlam_batch.py`)

New script implementing the plan's §13.7 pre-training checks. Runs **data-side
only by default** (no GPU, no weights; data-side imports do not touch the DCNv4
compiled ops).

It verifies:

- per-sample InterHand masks: `smplx_shape_valid/expr_valid/face_bbox_valid==0`;
  `L_Wrist/R_Wrist` pose valid `==0` with only the 15+15 finger joints pose-valid;
  `L_Wrist/R_Wrist` 3D-cam valid `==0`;
- per-hand wrist-relative coordinates: both `L_Wrist` and `R_Wrist` `joint_cam`
  must be `~=0` (own-hand origin) — the discriminator against the old
  shared-root convention — and valid finger magnitudes stay within `hand_3d_size`;
- BEDLAM `smplx_shape_valid==1` and source flags;
- InterHand/BEDLAM per-key tensor shapes match (mixed-batch collate safety);
- realized weighted-sampling ratio vs `cfg.trainset_3d_sample_prob` over several
  batches (uses the same `_seed_worker`).

Optional `--forward`:

- lazily builds the real `pytorch` model via `base.Tester._make_model` (frozen
  backbone remap + trained snapshot), runs an InterHand batch in test mode, and
  saves the input crops with the **model-predicted** left/right hand boxes drawn.
  This is the Stage-0 gate for "are the predicted InterHand hand boxes plausible?".

Usage:

```bash
# data-side checks
python tool/InterHand26M/check_interhand_bedlam_batch.py \
  --interhand_max_samples 256 --bedlam_max_samples 256 --num_batches 8 --batch_size 8

# also dump model-predicted hand boxes
python tool/InterHand26M/check_interhand_bedlam_batch.py --forward \
  --pretrained_model_path ../pretrained_models/osx_l.pth.tar \
  --continue_train_path <snapshot.pth.tar>
```

Validation:

- `python -m py_compile common/base.py tool/InterHand26M/check_interhand_bedlam_batch.py data/dataset.py` passed.
- Executed end-to-end on a Tesla T4 / 30 GB box (see §10 for the smoke-test run
  and the findings it surfaced). Data-side checks pass; peak RAM ~5.6 GB for the
  data-only path (`COCO()` parsing the 904 MB InterHand train JSON is the main
  cost), ~11 GB + 1.6 GB GPU for `--forward`.

## 10. Smoke Test + GT Hand Box Injection

### 10.1 Smoke Test Results

Ran `tool/InterHand26M/check_interhand_bedlam_batch.py` (data-side and `--forward`)
with small caps.

Passing:

- all InterHand per-sample masks (shape/expr/face=0, `L/R_Wrist` pose+3D valid=0,
  only finger joints pose-valid);
- per-hand wrist-relative coordinates confirmed (both wrists `joint_cam ~= 0`,
  valid finger magnitudes 0.14-0.18 m);
- BEDLAM `smplx_shape_valid=1`; InterHand/BEDLAM tensor shapes match;
- weighted sampling ratio (asserted at >= 64 drawn samples);
- full `pytorch` model forward runs end-to-end on InterHand.

Findings surfaced:

1. **Stage-0 box gate fails as the plan predicted.** The frozen top-down
   `box_net` outputs degenerate whole-frame (and out-of-bounds) hand boxes on
   InterHand hand-centric images, and does not distinguish left/right. This is
   the motivation for GT box injection below.
2. **`Tester`/`Demoer` resume `module.`-prefix loading — RESOLVED.** Earlier,
   `Tester._make_model` Step 2 matched a no-`module.`-prefix model against a
   snapshot whose keys are all `module.`-prefixed, loading **0** trained hand/face
   tensors (logged "成功覆盖了 0 个" yet printed "完美"). This is now handled by
   `_load_lightweight_trained_modules` (`common/base.py:193-200`), which tries
   both the `module.`-stripped and `module.`-added candidate key for every tensor.
   When first evaluating, just confirm the log "成功覆盖了 N 个" shows N > 0.

Check-script bugs fixed while running the smoke test: forward import
(`base`->`common.base`), `feat_dim` not set when `set_additional_args` is skipped
(osx_l needs 1024, not the class-default 768), relative pretrained-path
resolution, image de-normalization (dataset only does `/255`, no ImageNet
mean/std), and a minimum-sample guard on the ratio assertion.

### 10.2 GT Hand Box Injection

Files:

- `main/config.py`
- `main/model_core.py`

Problem:

- Finding (1) above: the frozen `box_net` cannot localize hands on InterHand, so
  the hand ROI crop (and the GT-2D-keypoint remapping that follows it) is driven
  by a degenerate whole-frame box.

Change:

- Added `cfg.inject_gt_hand_bbox = True`.
- In `Model.forward`, after `box_net` and before `restore_bbox`, for samples
  flagged `is_hand_only` with a valid GT hand bbox, the ROI-crop center/size are
  replaced with the GT `Xhand_bbox_center/size`:

```python
roi_lhand_center, roi_lhand_size = lhand_bbox_center, lhand_bbox_size
roi_rhand_center, roi_rhand_size = rhand_bbox_center, rhand_bbox_size
if cfg.inject_gt_hand_bbox and 'is_hand_only' in meta_info and 'lhand_bbox_center' in targets:
    hand_only = meta_info['is_hand_only'].view(-1) > 0
    l_use = (hand_only & (meta_info['lhand_bbox_valid'].view(-1) > 0)).view(-1, 1)
    r_use = (hand_only & (meta_info['rhand_bbox_valid'].view(-1) > 0)).view(-1, 1)
    roi_lhand_center = torch.where(l_use, targets['lhand_bbox_center'], roi_lhand_center)
    roi_lhand_size   = torch.where(l_use, targets['lhand_bbox_size'],   roi_lhand_size)
    roi_rhand_center = torch.where(r_use, targets['rhand_bbox_center'], roi_rhand_center)
    roi_rhand_size   = torch.where(r_use, targets['rhand_bbox_size'],   roi_rhand_size)
lhand_bbox = restore_bbox(roi_lhand_center, roi_lhand_size, ...).detach()
rhand_bbox = restore_bbox(roi_rhand_center, roi_rhand_size, ...).detach()
```

Design notes:

- box_net's own predictions (`lhand_bbox_center/size`, ...) are left untouched,
  so the (diagnostic) bbox loss is unchanged and box_net stays frozen.
- Both predicted and GT boxes are in `output_hm` space and go through the same
  `restore_bbox` (aspect-ratio + 2.0x extension), so the injected crop matches
  the input distribution the hand nets were pretrained on.
- Gated on `is_hand_only`, so BEDLAM / EHF / other datasets are unaffected.
- An invalid hand in a single-hand sample keeps the predicted box; its losses are
  masked anyway.
- Consistency note: this also makes the hand 2D-keypoint target remapping (which
  remaps into the chosen box) self-consistent with the actual crop content.

Verification (no model needed, `restore_bbox` on CPU):

- bbox-valid hand == keypoint-valid hand for every sampled item (no flip bug);
- 20/20 sampled InterHand items have **100%** of the valid hand's visible
  keypoints inside the GT-injected box;
- `--forward` visual dump (now overlaying GT keypoints) confirms tight boxes on
  partial-hand images and full-frame boxes on hand-fills-frame images.

Caveat:

- This bypasses box_net rather than fixing it; at inference time without a GT box
  (real deployment) InterHand would fall back to the degenerate predicted box.
  The training goal is to improve the hand position net / decoder / regressor,
  for which the injected crop path is sufficient.

## 11. Suggested Smoke Run

Use a capped one-epoch run before long training:

```bash
cd /workspace/myosx/main
python train.py \
  --gpu_ids 0 \
  --decoder_setting pytorch \
  --lr 5e-5 \
  --lr_mult 0.1 \
  --end_epoch 1 \
  --phase1_epochs 1 \
  --train_batch_size 8 \
  --num_thread 2 \
  --bedlam_max_samples 64 \
  --interhand_max_samples 64 \
  --pretrained_model_path ../pretrained_models/osx_l.pth.tar \
  --exp_name output/debug_interhand_bedlam
```

Expected early checks:

- Log shows weighted dataset sampling ratio.
- Batch logs include nonzero `batch_interhand` and `batch_bedlam` over time.
- Phase 1 trainable modules include `hand_position_net`, `hand_decoder`, and `hand_regressor`.
- BEDLAM hand poses are converted once and contain `flat_hand_mean_converted=True` in memory.


## 12. Status Summary and Next Steps

### 12.1 Resolved

Original-plan blockers and review items that are now implemented and verified:

- [RESOLVED] Active model entry forced to `pytorch` (`cfg.decoder_setting='pytorch'`
  default + hard guard in `main/train.py`). §1, §7.
- [RESOLVED] Phase boundary fixed so `hand_regressor` actually trains during
  hand finetuning (`phase1_train_hand_regressor`, phase logic in
  `model_core.set_training_phase` + `train._configure_training_phase`). §6, §7.
- [RESOLVED] InterHand 3D targets are per-hand wrist-relative; `L_Wrist/R_Wrist`
  3D-cam valid = 0. Verified: both wrists `joint_cam ~= 0`, fingers 0.14-0.18 m,
  20/20 samples consistent. §4, §10.1.
- [RESOLVED] InterHand wrist pose supervision disabled; only 15+15 finger joints
  pose-valid; shape/expr/face masks = 0. Verified by the batch checker. §4, §10.1.
- [RESOLVED] Numeric dataset-source flags in both loaders, identical dict keys,
  collate-safe (shapes confirmed equal). §4, §5.1, §10.1.
- [RESOLVED] BEDLAM `flat_hand_mean` compatibility conversion (+ double-convert
  guard, .npz/.pkl/getitem coverage). §5.2.
- [RESOLVED] Weighted dataset sampling (`MultipleDatasets(sample_prob=...)` wired
  via `cfg.trainset_3d_sample_prob`); ratio asserted in the checker. §2, §3, §10.1.
- [RESOLVED] Per-worker numpy/random seeding for correct weighted sampling. §9.1.
- [RESOLVED] Stage-0 batch/mask checker script exists and runs end-to-end. §9.2.
- [RESOLVED] Stage-0 predicted-hand-box visualization gate (`--forward`). §9.2, §10.1.
- [RESOLVED] Degenerate frozen-`box_net` hand boxes on InterHand mitigated via GT
  hand box injection for the ROI crop. §10.2.
- [RESOLVED] `Tester`/`Demoer` `module.`-prefix snapshot loading: handled by
  `_load_lightweight_trained_modules` (`common/base.py:193-200`), which tries both
  the `module.`-stripped and `module.`-added key per tensor. §10.1 finding (2).
- [RESOLVED] Rotation-matrix hand pose loss implemented (default OFF):
  `RotMatParamLoss` in `common/nets/loss.py` (axis-angle -> rotmat via
  `batch_rodrigues`, masked L1), wired in `main/model_core.py` as an ADDITIVE,
  hand-finger-only term (`smpl_x.orig_joint_part['lhand']+['rhand']` = 30 joints,
  masked by `smplx_pose_valid`) under `cfg.use_hand_rotmat_pose_loss` /
  `--use_hand_rotmat_pose_loss` (+ `--hand_rotmat_pose_loss_weight`). Unit-tested
  (incl. 2π rotation-invariance: rotmat loss ~0 where axis-angle L1 ~2.1). Not yet
  exercised in a real training run.

### 12.2 Open / Pending

- [PARTIAL] Full smoke/validation run of the real `main/train.py`. Data loading,
  the gradient-flow check, and the first training iterations now run. The earlier
  exit-137 OOM pressure (InterHand permanently retaining the full raw
  COCO/joint/MANO annotations) was removed — they are freed after `load_data` —
  and corrupt-image crashes in BOTH loaders are caught by a bounded resample
  fallback in `__getitem__`. Still pending: a run past the epoch2-itr~429 hand
  heatmap divergence point to validate the B-class hand-img fix (`continue.txt` §三).
- [OPEN] `test.py:223` hard-stops at `if itr >= 50: break` (partial eval); remove
  or make configurable before real evaluation.
- [PENDING] Stage 1-4 training per the plan (BEDLAM warmup -> 60:40 hand finetune
  -> 50:50 stabilize -> optional polish), with per-dataset loss logging and the
  every-epoch visual panel.
- [DEFERRED] Weighted sampler vs `itr_per_epoch` decoupling (epoch length still
  follows dataset-length sum). Acceptable for now.

### 12.3 Recommended Next Step

On the high-memory server, in order:

1. Run the data-side checker (`--interhand_max_samples 256 --bedlam_max_samples 256`)
   as a regression gate.
2. Run the capped real smoke train (§11) and confirm: no NaN, weighted ratio in
   logs, Phase 1 trains `hand_position_net/hand_decoder/hand_regressor`, gradient
   flow check passes, GT-box injection active for InterHand.
3. When first evaluating, confirm the snapshot actually loaded (log "成功覆盖了
   N 个" with N > 0) and remove the `test.py:223` 50-batch debug cap.
4. Proceed to Stage 1 (BEDLAM-only warmup) -> Stage 2 (60:40), monitoring the
   §8.3 validation gates (InterHand hand loss down, BEDLAM full-body loss not up
   by >5-10%, no left/right swap).
