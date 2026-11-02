"""Source mapping for the extracted production path."""

import subprocess
from pathlib import Path


SOURCE_HAWOR_COMMIT = "579253db1787b306161e81444170178cd6628e37"
SOURCE_FILES = {
    "detection": "scripts/ego4d_sam3_hawor/pipeline.py:4708 save_yolo_detector_area_max_bboxes",
    "payload": "scripts/ego4d_sam3_hawor/detection_payload.py",
    "sam3_mask": "scripts/ego4d_sam3_hawor/sam3_point.py",
    "sam3_helpers": "scripts/ego4d_sam3_hawor/sam_anchor.py and sam_repropagate.py",
    "ego_gate": "scripts/ego4d_sam3_hawor/ego_candidate_score.py",
}


def _git_output(repo_root: Path, *args: str) -> str:
    """Run Git compatibly with old compute-node clients; provenance is optional."""
    try:
        return subprocess.check_output(
            ["git", *args],
            cwd=repo_root,
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return ""


def git_commit(repo_root: Path) -> str:
    return _git_output(repo_root, "rev-parse", "HEAD") or "unknown"


def git_dirty(repo_root: Path) -> bool:
    return bool(_git_output(repo_root, "status", "--porcelain"))
