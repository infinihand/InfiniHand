"""Command line interface for the standalone hand mask pipeline."""

import argparse
from pathlib import Path

from .config import load_config


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="HaWoR YOLO hand detection + SAM3 point-prompt masks"
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    def add_run_arguments(subparser):
        subparser.add_argument("--config", type=Path, required=True)
        subparser.add_argument("--output-dir", type=Path, required=True)
        subparser.add_argument("--detector-weight", type=Path, required=True)
        subparser.add_argument("--sam3-checkpoint", type=Path, required=True)
        subparser.add_argument(
            "--sam3-src", type=Path, default=Path("/path/to/infinihand/third_party/sam3")
        )
        subparser.add_argument("--device", default="cuda")
        subparser.add_argument("--max-frames", type=int)
        subparser.add_argument("--overwrite", action="store_true")

    video = subparsers.add_parser("video", help="run on one input video")
    add_run_arguments(video)
    video.add_argument("--video", type=Path, required=True)

    frames = subparsers.add_parser("frames", help="run on an ordered image folder")
    add_run_arguments(frames)
    frames.add_argument("--frames-dir", type=Path, required=True)

    visualize = subparsers.add_parser(
        "visualize", help="rerender overlay from an existing output directory"
    )
    visualize.add_argument("--config", type=Path, required=True)
    visualize.add_argument("--output-dir", type=Path, required=True)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    config = load_config(args.config)
    from .pipeline import run_pipeline, visualize_existing

    if args.command == "visualize":
        visualize_existing(args.output_dir, config)
        return 0
    if args.max_frames is not None:
        config["max_frames"] = int(args.max_frames)
    input_type = args.command
    input_path = args.video if args.command == "video" else args.frames_dir
    run_pipeline(
        input_type=input_type,
        input_path=input_path,
        output_dir=args.output_dir,
        config=config,
        detector_weight=args.detector_weight,
        sam3_checkpoint=args.sam3_checkpoint,
        sam3_src=args.sam3_src,
        device=args.device,
        overwrite=args.overwrite,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
