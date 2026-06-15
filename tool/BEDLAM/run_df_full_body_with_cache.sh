#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

PYTHON_BIN="${PYTHON_BIN:-python}"
BEDLAM_CODE_ROOT="${BEDLAM_CODE_ROOT:-/workspace/BEDLAM}"
BEDLAM_DATASET_ROOT="${BEDLAM_DATASET_ROOT:-/workspace/BEDLAM_Dataset}"
SMPLX_GT_FOLDER="${SMPLX_GT_FOLDER:-${BEDLAM_DATASET_ROOT}/smplx_gt/neutral_ground_truth_motioninfo}"
BEDLAM_PROCESSED_LABELS="${BEDLAM_PROCESSED_LABELS:-${BEDLAM_DATASET_ROOT}/processed_labels}"
BEDLAM_PROCESS_FPS="${BEDLAM_PROCESS_FPS:-1}"
BEDLAM_IMAGE_FORMAT="${BEDLAM_IMAGE_FORMAT:-.jpg}"
BEDLAM_DEVICE="${BEDLAM_DEVICE:-auto}"

"${PYTHON_BIN}" "${PROJECT_ROOT}/tool/BEDLAM/df_full_body_with_cache.py" \
  --bedlam-code-root "${BEDLAM_CODE_ROOT}" \
  --img_folder "${BEDLAM_DATASET_ROOT}" \
  --smplx_gt_folder "${SMPLX_GT_FOLDER}" \
  --output_folder "${BEDLAM_PROCESSED_LABELS}" \
  --fps "${BEDLAM_PROCESS_FPS}" \
  --img_format "${BEDLAM_IMAGE_FORMAT}" \
  --device "${BEDLAM_DEVICE}" \
  "$@"
