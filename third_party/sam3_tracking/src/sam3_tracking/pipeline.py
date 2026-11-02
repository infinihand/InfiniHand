"""End-to-end standalone hand detection and SAM3 mask pipeline."""

import json
import os
import shutil
import sys
import time
from pathlib import Path

import torch

from .detection import detect_hands, detection_metrics
from .postprocess import restore_masks
from .provenance import (
    SOURCE_FILES,
    SOURCE_HAWOR_COMMIT,
    git_commit,
    git_dirty,
)
from .sam3_input import prepare_sam3_input
from .video_io import (
    extract_video_frames,
    prepare_frame_folder,
    write_frame_manifest,
)
from .visualization import write_visualization


REPO_ROOT = Path(__file__).resolve().parents[2]


def write_json(path: Path, payload: dict) -> None:
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def point_prompt_argv(
    config: dict,
    sam3_segment_dir: Path,
    sam3_raw_dir: Path,
    checkpoint_path: Path,
    standalone_commit: str,
) -> list[str]:
    argv = [
        "--segment-dir",
        str(sam3_segment_dir),
        "--output-dir",
        str(sam3_raw_dir),
        "--checkpoint-path",
        str(checkpoint_path),
        "--sam-version",
        str(config["sam3_version"]),
        "--fps",
        str(config["output_fps"]),
        "--score-thresh",
        str(config["sam3_score_thresh"]),
        "--bottom-y2-frac",
        str(config["sam3_bottom_y2_frac"]),
        "--window-sec",
        str(config["sam3_window_sec"]),
        "--point-expand",
        str(config["sam3_point_expand"]),
        "--prompt-geometry",
        str(config["sam3_prompt_geometry"]),
        "--neg-below-count",
        str(config["sam3_neg_below_count"]),
        "--neg-below-frac",
        str(config["sam3_neg_below_frac"]),
        "--sam-output-thresh",
        str(config["sam3_output_thresh"]),
        "--default-output-prob-thresh",
        str(config["sam3_default_output_prob_thresh"]),
        "--prompt-text",
        str(config["sam3_prompt_text"]),
        "--tracking-mode",
        str(config["sam3_tracking_mode"]),
        "--init-mode",
        str(config["sam3_init_mode"]),
        "--propagation-scheme",
        str(config["sam3_propagation_scheme"]),
        "--mask-archive",
        str(config["sam3_mask_archive"]),
        "--seed-window-ratio-min",
        str(config["sam3_seed_window_ratio_min"]),
        "--git-commit",
        standalone_commit,
    ]
    if config["sam3_ego_gate_enabled"]:
        argv.append("--ego-gate-enabled")
        for key in [
            "ego_gate_size_ratio_min",
            "ego_gate_size_ratio_good",
            "ego_gate_bottom_ratio_min",
            "ego_gate_bottom_ratio_good",
            "ego_gate_area_weight",
            "ego_gate_bottom_weight",
            "ego_gate_keep_tau",
            "ego_gate_flag_tau",
            "ego_gate_margin_tau",
        ]:
            argv.extend(["--" + key.replace("_", "-"), str(config[key])])
    for config_key, flag in [
        ("sam3_exec_micro_opts", "--exec-micro-opts"),
        ("sam3_offload_video", "--offload-video"),
        ("sam3_offload_state", "--offload-state"),
        ("sam3_chunk_trim", "--chunk-trim"),
        ("sam3_dedup_pos_enc", "--dedup-pos-enc"),
        ("sam3_close_empty_cache", "--close-empty-cache"),
    ]:
        if config[config_key]:
            argv.append(flag)
    return argv


def run_pipeline(
    input_type: str,
    input_path: Path,
    output_dir: Path,
    config: dict,
    detector_weight: Path,
    sam3_checkpoint: Path,
    sam3_src: Path,
    device: str,
    overwrite: bool,
) -> dict:
    if overwrite and output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True)
    standalone_commit = git_commit(REPO_ROOT)
    run_config = {
        **config,
        "input_type": input_type,
        "input_path": str(input_path.resolve()),
        "output_dir": str(output_dir.resolve()),
        "detector_weight": str(detector_weight.resolve()),
        "sam3_checkpoint": str(sam3_checkpoint.resolve()),
        "sam3_src": str(sam3_src.resolve()),
        "sam3_git_commit": git_commit(sam3_src),
        "sam3_git_dirty": git_dirty(sam3_src),
        "device": device,
        "standalone_git_commit": standalone_commit,
        "standalone_git_dirty": git_dirty(REPO_ROOT),
        "source_hawor_commit": SOURCE_HAWOR_COMMIT,
        "source_files": SOURCE_FILES,
    }
    write_json(output_dir / "run_config.json", run_config)
    wandb_run = None
    if str(config["wandb_mode"]).lower() != "disabled":
        import wandb

        wandb_run = wandb.init(
            project=str(config["wandb_project"]),
            name=output_dir.name,
            mode=str(config["wandb_mode"]),
            dir=str(output_dir),
            config=run_config,
        )
    timings = {}
    frame_dir = output_dir / "frames"
    started = time.perf_counter()
    if input_type == "video":
        frame_manifest = extract_video_frames(
            input_path,
            frame_dir,
            config["output_fps"],
            config["jpg_quality"],
            config["max_frames"],
        )
    else:
        assert input_type == "frames", input_type
        frame_manifest = prepare_frame_folder(
            input_path,
            frame_dir,
            config["output_fps"],
            config["jpg_quality"],
            config["max_frames"],
        )
    timings["frame_prepare_seconds"] = time.perf_counter() - started
    write_frame_manifest(output_dir / "frame_manifest.json", frame_manifest)
    frame_paths = sorted(frame_dir.glob("*.jpg"))
    image_size_hw = (frame_manifest["height"], frame_manifest["width"])

    started = time.perf_counter()
    detection_payload = detect_hands(
        frame_paths=frame_paths,
        image_size_hw=image_size_hw,
        detector_weight=detector_weight,
        output_path=output_dir / "detection" / "bboxes_2d.pth",
        detections_jsonl=output_dir / "detections.jsonl",
        device=device,
        confidence=config["detector_confidence"],
        batch_size=config["detector_batch_size"],
        imgsz=config["detector_imgsz"],
        video_id=input_path.stem,
    )
    timings["detection_seconds"] = time.perf_counter() - started
    metrics = detection_metrics(detection_payload, len(frame_paths))

    started = time.perf_counter()
    sam3_segment_dir = output_dir / "sam3_input"
    resize_metadata = prepare_sam3_input(
        frame_paths,
        detection_payload,
        sam3_segment_dir,
        image_size_hw,
        config["sam3_input_max_side"],
        config["jpg_quality"],
    )
    timings["sam3_input_prepare_seconds"] = time.perf_counter() - started

    if config["stage_events"]:
        os.environ["HAND_SAM3_STAGE_EVENTS"] = "1"
        os.environ["HAND_SAM3_STAGE_EVENTS_PATH"] = str(
            output_dir / "stage_events.jsonl"
        )
    os.environ["SAM3_SRC"] = str(sam3_src.resolve())
    sys.path.insert(0, str(sam3_src.resolve()))
    from . import sam3_point

    sam3_raw_dir = output_dir / "sam3_raw"
    point_args = sam3_point.parse_args(
        point_prompt_argv(
            config,
            sam3_segment_dir,
            sam3_raw_dir,
            sam3_checkpoint,
            standalone_commit,
        )
    )
    started = time.perf_counter()
    sam3_point.run_point_prompt(point_args)
    timings["sam3_seconds"] = time.perf_counter() - started

    started = time.perf_counter()
    mask_metrics = restore_masks(
        sam3_raw_dir,
        output_dir,
        len(frame_paths),
        image_size_hw,
        tuple(resize_metadata["sam3_image_size_hw"]),
        config["sam3_mask_archive"],
    )
    timings["mask_restore_seconds"] = time.perf_counter() - started
    metrics.update(mask_metrics)
    if config["write_visualization"]:
        started = time.perf_counter()
        write_visualization(
            frame_paths,
            output_dir / "masks",
            detection_payload,
            output_dir,
            config["output_fps"],
            config["visualization_alpha"],
            config["contact_frames"],
        )
        timings["visualization_seconds"] = time.perf_counter() - started
    timings["total_seconds"] = sum(timings.values())
    result = {
        "status": "ok",
        "num_frames": len(frame_paths),
        "metrics": metrics,
        "timings": timings,
        "outputs": {
            "detections": "detection/bboxes_2d.pth",
            "detection_jsonl": "detections.jsonl",
            "masks": "masks/",
            "mask_jsonl": "masks.jsonl",
            "sam3_raw": "sam3_raw/",
            "overlay": "overlay.mp4" if config["write_visualization"] else None,
            "contact_sheet": "contact_sheet.jpg"
            if config["write_visualization"]
            else None,
        },
    }
    write_json(output_dir / "summary.json", result)
    if wandb_run is not None:
        wandb_run.log({**metrics, **timings})
        wandb_run.finish()
    return result


def visualize_existing(output_dir: Path, config: dict) -> dict:
    frame_manifest = json.loads(
        (output_dir / "frame_manifest.json").read_text(encoding="utf-8")
    )
    frame_paths = sorted((output_dir / "frames").glob("*.jpg"))
    detection_payload = torch.load(
        output_dir / "detection" / "bboxes_2d.pth",
        map_location="cpu",
        weights_only=False,
    )
    return write_visualization(
        frame_paths,
        output_dir / "masks",
        detection_payload,
        output_dir,
        frame_manifest["output_fps"],
        config["visualization_alpha"],
        config["contact_frames"],
    )
