#!/usr/bin/env bash
set -euo pipefail

if [ "$#" -lt 2 ]; then
  echo "usage: $0 FRAMES_DIR OUTPUT_DIR [extra hand-sam3-mask args]" >&2
  exit 2
fi

PROJECT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
PYTHON_BIN=${PYTHON_BIN:-$PROJECT_DIR/.venv/bin/python}
INPUT_FRAMES=$1
OUTPUT_DIR=$2
shift 2

cd "$PROJECT_DIR"
exec "$PYTHON_BIN" -m hand_detection_sam3_mask frames \
  --config configs/default.yaml \
  --frames-dir "$INPUT_FRAMES" \
  --output-dir "$OUTPUT_DIR" \
  --detector-weight weights/detector.pt \
  --sam3-checkpoint checkpoints/facebook_sam3/sam3.pt \
  --sam3-src third_party/sam3 \
  "$@"
