#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
PYTHON_BIN=${PYTHON_BIN:-python3.12}
TORCH_INDEX_URL=${TORCH_INDEX_URL:-https://download.pytorch.org/whl/cu128}
SAM3_GIT_URL=${SAM3_GIT_URL:-https://github.com/facebookresearch/sam3.git}
SAM3_GIT_REF=${SAM3_GIT_REF:-main}
SAM3_SOURCE_MODE=${SAM3_SOURCE_MODE:-bundled}
SAM3_SRC=${SAM3_SRC:-$PROJECT_DIR/third_party/sam3}

cd "$PROJECT_DIR"
"$PYTHON_BIN" -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install torch==2.10.0 torchvision --index-url "$TORCH_INDEX_URL"
if [ "$SAM3_SOURCE_MODE" = "clone" ]; then
  [ ! -e "$SAM3_SRC" ]
  git clone --branch "$SAM3_GIT_REF" --depth 1 "$SAM3_GIT_URL" "$SAM3_SRC"
else
  [ "$SAM3_SOURCE_MODE" = "bundled" ]
  [ -f "$SAM3_SRC/pyproject.toml" ]
fi
python -m pip install -e "$SAM3_SRC"
python -m pip install -e '.[dev]'
python -m hand_detection_sam3_mask --help
