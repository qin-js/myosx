#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

PYTHON_BIN="${PYTHON_BIN:-python}"
BEDLAM_CODE_ROOT="${BEDLAM_CODE_ROOT:-/workspace/BEDLAM}"
BEDLAM_DATASET_ROOT="${BEDLAM_DATASET_ROOT:-/workspace/BEDLAM_Dataset}"
SMPLX_GT_FOLDER="${SMPLX_GT_FOLDER:-${BEDLAM_DATASET_ROOT}/smplx_gt/neutral_ground_truth_motioninfo}"
BEDLAM_PRECHECK_OUTPUT="${BEDLAM_PRECHECK_OUTPUT:-${BEDLAM_DATASET_ROOT}/precheck_vis}"
BEDLAM_PROCESS_FPS="${BEDLAM_PROCESS_FPS:-2}"
BEDLAM_IMAGE_FORMAT="${BEDLAM_IMAGE_FORMAT:-.jpg}"
BEDLAM_PRECHECK_SAMPLES="${BEDLAM_PRECHECK_SAMPLES:-24}"

"${PYTHON_BIN}" "${PROJECT_ROOT}/tool/BEDLAM/precheck_df_full_body.py" \
  --bedlam-code-root "${BEDLAM_CODE_ROOT}" \
  --img-folder "${BEDLAM_DATASET_ROOT}" \
  --smplx-gt-folder "${SMPLX_GT_FOLDER}" \
  --output-dir "${BEDLAM_PRECHECK_OUTPUT}" \
  --fps "${BEDLAM_PROCESS_FPS}" \
  --img-format "${BEDLAM_IMAGE_FORMAT}" \
  --num-samples "${BEDLAM_PRECHECK_SAMPLES}" \
  "$@"
