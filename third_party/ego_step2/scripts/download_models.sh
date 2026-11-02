#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
PYTHON_BIN=${PYTHON_BIN:-$PROJECT_DIR/.venv/bin/python}

cd "$PROJECT_DIR"
exec "$PYTHON_BIN" -m hand_detection_sam3_mask.download_models "$@"
