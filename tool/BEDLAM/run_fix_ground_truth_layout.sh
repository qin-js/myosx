#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

PYTHON_BIN="${PYTHON_BIN:-python}"
BEDLAM_DATASET_ROOT="${BEDLAM_DATASET_ROOT:-/workspace/BEDLAM_Dataset}"

"${PYTHON_BIN}" "${PROJECT_ROOT}/tool/BEDLAM/fix_ground_truth_layout.py" \
  --root "${BEDLAM_DATASET_ROOT}" \
  "$@"
