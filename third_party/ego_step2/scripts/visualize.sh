#!/usr/bin/env bash
set -euo pipefail

if [ "$#" -ne 1 ]; then
  echo "usage: $0 OUTPUT_DIR" >&2
  exit 2
fi

PROJECT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
PYTHON_BIN=${PYTHON_BIN:-$PROJECT_DIR/.venv/bin/python}

cd "$PROJECT_DIR"
exec "$PYTHON_BIN" -m hand_detection_sam3_mask visualize \
  --config configs/default.yaml \
  --output-dir "$1"
