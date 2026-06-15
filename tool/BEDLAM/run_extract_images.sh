#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

PYTHON_BIN="${PYTHON_BIN:-python}"
BEDLAM_DATASET_ROOT="${BEDLAM_DATASET_ROOT:-/workspace/BEDLAM_Dataset}"
BEDLAM_EXTRACT_FPS="${BEDLAM_EXTRACT_FPS:-2}"
BEDLAM_IMAGE_EXT="${BEDLAM_IMAGE_EXT:-jpg}"
BEDLAM_JPG_QUALITY="${BEDLAM_JPG_QUALITY:-95}"
BEDLAM_WORKERS="${BEDLAM_WORKERS:-8}"

"${PYTHON_BIN}" "${PROJECT_ROOT}/tool/BEDLAM/video2image.py" \
  --root "${BEDLAM_DATASET_ROOT}" \
  --fps "${BEDLAM_EXTRACT_FPS}" \
  --ext "${BEDLAM_IMAGE_EXT}" \
  --jpg-quality "${BEDLAM_JPG_QUALITY}" \
  --workers "${BEDLAM_WORKERS}" \
  "$@"
