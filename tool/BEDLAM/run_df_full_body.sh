#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-python}"
BEDLAM_CODE_ROOT="${BEDLAM_CODE_ROOT:-/workspace/BEDLAM}"
BEDLAM_DATASET_ROOT="${BEDLAM_DATASET_ROOT:-/workspace/BEDLAM_Dataset}"
SMPLX_GT_FOLDER="${SMPLX_GT_FOLDER:-${BEDLAM_DATASET_ROOT}/smplx_gt/neutral_ground_truth_motioninfo}"
BEDLAM_PROCESSED_LABELS="${BEDLAM_PROCESSED_LABELS:-${BEDLAM_DATASET_ROOT}/processed_labels}"
BEDLAM_PROCESS_FPS="${BEDLAM_PROCESS_FPS:-2}"
BEDLAM_IMAGE_FORMAT="${BEDLAM_IMAGE_FORMAT:-.jpg}"

cd "${BEDLAM_CODE_ROOT}/data_processing"

"${PYTHON_BIN}" df_full_body.py \
  --img_folder "${BEDLAM_DATASET_ROOT}" \
  --smplx_gt_folder "${SMPLX_GT_FOLDER}" \
  --output_folder "${BEDLAM_PROCESSED_LABELS}" \
  --fps "${BEDLAM_PROCESS_FPS}" \
  --img_format "${BEDLAM_IMAGE_FORMAT}" \
  "$@"