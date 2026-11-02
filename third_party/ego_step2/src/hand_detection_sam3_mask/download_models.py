"""Download the public detector and gated SAM3 checkpoint."""

import argparse
import urllib.request
from pathlib import Path

from huggingface_hub import hf_hub_download


DETECTOR_URL = "https://huggingface.co/spaces/rolpotamias/WiLoR/resolve/main/pretrained_models/detector.pt"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--detector-output", type=Path, default=Path("/path/to/weights/detector.pt")
    )
    parser.add_argument(
        "--sam3-output-dir", type=Path, default=Path("/path/to/checkpoints/facebook_sam3")
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    args.detector_output.parent.mkdir(parents=True, exist_ok=True)
    urllib.request.urlretrieve(DETECTOR_URL, args.detector_output)
    args.sam3_output_dir.mkdir(parents=True, exist_ok=True)
    for filename in ["sam3.pt", "config.json"]:
        hf_hub_download(
            repo_id="facebook/sam3",
            filename=filename,
            local_dir=args.sam3_output_dir,
            token=True,
        )
    print(args.detector_output)
    print(args.sam3_output_dir / "sam3.pt")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
