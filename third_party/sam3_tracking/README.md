# Optional SAM3 hand tracking

This component supplies YOLO detection and SAM3 point-prompt tracking for `infinihand.infer --mask-source sam3-tracking`. The default InfiniHand mask head does not need this component.

Install the requirements and SAM3 in a separate CUDA environment:

```bash
python -m pip install -r /path/to/InfiniHand/third_party/sam3_tracking/requirements.txt
python -m pip install -e /path/to/InfiniHand/third_party/sam3_tracking
python -m pip install -e /path/to/sam3
```

Obtain the detector weights and gated SAM3 checkpoint separately. Detector classes must be `0=left` and `1=right`. No source models or checkpoints are bundled here. Point `--sam3-python` at this environment and pass `--sam3-source`, `--sam3-checkpoint`, and `--detector` to the InfiniHand inference command.

Standalone frame processing:

```bash
python -m sam3_tracking frames \
  --frames-dir /path/to/frames --output-dir /path/to/output \
  --config /path/to/InfiniHand/third_party/sam3_tracking/configs/default.yaml \
  --detector-weight /path/to/weights/detector.pt \
  --sam3-checkpoint /path/to/weights/sam3.pt --sam3-src /path/to/sam3
```

The output contains masks, per-frame boxes and `masks.jsonl`. Frame dimensions, side labels and mask/box correspondence are preserved. Use a new output directory for each sequence. The default configuration uses joint tracking, point initialization and five-second windows.

Source attribution and upstream terms are retained in [NOTICE.md](NOTICE.md) and [the source map](docs/SOURCE_MAP.md). Runtime settings are listed in `configs/default.yaml`.
