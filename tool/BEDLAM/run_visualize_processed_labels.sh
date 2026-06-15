#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

PYTHON_BIN="${PYTHON_BIN:-python}"
BEDLAM_CODE_ROOT="${BEDLAM_CODE_ROOT:-/workspace/BEDLAM}"
BEDLAM_DATASET_ROOT="${BEDLAM_DATASET_ROOT:-/workspace/BEDLAM_Dataset}"
BEDLAM_PROCESSED_LABELS="${BEDLAM_PROCESSED_LABELS:-${BEDLAM_DATASET_ROOT}/processed_labels}"
BEDLAM_VIS_OUTPUT="${BEDLAM_VIS_OUTPUT:-${BEDLAM_DATASET_ROOT}/vis_check}"
BEDLAM_VIS_SAMPLES="${BEDLAM_VIS_SAMPLES:-24}"
BEDLAM_VIS_PATTERN="${BEDLAM_VIS_PATTERN:-auto}"
BEDLAM_VIS_RENDERER="${BEDLAM_VIS_RENDERER:-myosx}"
BEDLAM_JPG_QUALITY="${BEDLAM_JPG_QUALITY:-95}"

"${PYTHON_BIN}" "${PROJECT_ROOT}/tool/BEDLAM/visualize_processed_labels.py" \
  --project-root "${PROJECT_ROOT}" \
  --bedlam-code-root "${BEDLAM_CODE_ROOT}" \
  --img-root "${BEDLAM_DATASET_ROOT}" \
  --label-dir "${BEDLAM_PROCESSED_LABELS}" \
  --output-dir "${BEDLAM_VIS_OUTPUT}" \
  --pattern "${BEDLAM_VIS_PATTERN}" \
  --num-samples "${BEDLAM_VIS_SAMPLES}" \
  --renderer "${BEDLAM_VIS_RENDERER}" \
  --jpg-quality "${BEDLAM_JPG_QUALITY}" \
  "$@"
