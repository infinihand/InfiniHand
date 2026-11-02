#!/usr/bin/env bash
set -euo pipefail

if [ "$#" -lt 2 ]; then
  echo "usage: $0 INPUT.mp4 OUTPUT_DIR [extra hand-sam3-mask args]" >&2
  exit 2
fi

PROJECT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
PYTHON_BIN=${PYTHON_BIN:-$PROJECT_DIR/.venv/bin/python}
INPUT_VIDEO=$1
OUTPUT_DIR=$2
shift 2

cd "$PROJECT_DIR"
exec "$PYTHON_BIN" -m hand_detection_sam3_mask video \
  --config configs/default.yaml \
  --video "$INPUT_VIDEO" \
  --output-dir "$OUTPUT_DIR" \
  --detector-weight weights/detector.pt \
  --sam3-checkpoint checkpoints/facebook_sam3/sam3.pt \
  --sam3-src third_party/sam3 \
  "$@"
