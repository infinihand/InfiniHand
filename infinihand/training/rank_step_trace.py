from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any, Dict


TRACE_RECORD_KEYS = (
    "step",
    "finetune_step",
    "fetch_ms",
    "fetch_attempts",
    "pre_step_ms",
    "forward_ms",
    "loss_ms",
    "decision_ms",
    "backward_ms",
    "optimizer_ms",
    "step_ms",
    "iteration_ms",
    "objective",
    "loss_grouped_objective",
    "loss_stage1_weighted",
    "loss_stage2_weighted",
    "loss_camera_hand_weighted",
    "loss_world_hand_weighted",
    "loss_detection_weighted",
    "loss_pose_weighted",
    "outlier_skipped",
    "loss_outlier_ratio",
    "grad_norm",
    "lingbot_grad_norm",
    "valid_hands",
    "lr",
    "lingbot_lr",
)


def _jsonable(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "cpu"):
        value = value.cpu()
    if hasattr(value, "tolist"):
        return value.tolist()
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    return str(value)


def extract_batch_identity(batch: Dict[str, Any]) -> Dict[str, Any]:
    """Return the dataset, scene, and frame IDs for a rank-local batch."""
    return {
        "dataset_name": _jsonable(batch.get("dataset_name")),
        "seq_name": _jsonable(batch.get("seq_name")),
        "frame_ids": _jsonable(batch.get("ids")),
    }


def _proc_value_gb(path: str, key: str) -> float | None:
    try:
        with open(path, "r", encoding="utf-8") as handle:
            for line in handle:
                if line.startswith(key + ":"):
                    fields = line.split()
                    return float(fields[1]) / (1024.0**2)
    except (FileNotFoundError, OSError, ValueError, IndexError):
        return None
    return None


class RankStepTracer:
    """Write one line-buffered sample/timing/resource record per rank and step."""

    def __init__(
        self,
        output_dir: str | Path,
        *,
        rank: int,
        local_rank: int,
        world_size: int,
        flush_interval: int = 1,
    ) -> None:
        trace_dir = Path(output_dir) / "rank_traces"
        trace_dir.mkdir(parents=True, exist_ok=True)
        self.path = trace_dir / f"rank_{int(rank):03d}.jsonl"
        self.rank = int(rank)
        self.local_rank = int(local_rank)
        self.world_size = int(world_size)
        self.hostname = os.environ.get("HOSTNAME", "")
        self.flush_interval = max(1, int(flush_interval))
        self._records_since_flush = 0
        self._handle = self.path.open("a", encoding="utf-8", buffering=1)
        self._previous_wall_time = time.perf_counter()
        self._previous_process_time = time.process_time()

    def _resource_snapshot(self, torch_module: Any, device: Any) -> Dict[str, Any]:
        now_wall = time.perf_counter()
        now_process = time.process_time()
        wall_delta = max(now_wall - self._previous_wall_time, 1e-9)
        process_delta = max(now_process - self._previous_process_time, 0.0)
        self._previous_wall_time = now_wall
        self._previous_process_time = now_process
        try:
            load1, load5, load15 = os.getloadavg()
        except OSError:
            load1 = load5 = load15 = None
        snapshot: Dict[str, Any] = {
            "cpu_process_percent": 100.0 * process_delta / wall_delta,
            "cpu_process_rss_gb": _proc_value_gb(
                "/proc/self/status", "VmRSS"
            ),
            "cpu_process_peak_rss_gb": _proc_value_gb(
                "/proc/self/status", "VmHWM"
            ),
            "node_memory_available_gb": _proc_value_gb(
                "/proc/meminfo", "MemAvailable"
            ),
            "node_load1": load1,
            "node_load5": load5,
            "node_load15": load15,
        }
        if torch_module.cuda.is_available():
            free_bytes, total_bytes = torch_module.cuda.mem_get_info(device)
            snapshot.update(
                {
                    "gpu_memory_allocated_gb": (
                        torch_module.cuda.memory_allocated(device) / (1024.0**3)
                    ),
                    "gpu_memory_reserved_gb": (
                        torch_module.cuda.memory_reserved(device) / (1024.0**3)
                    ),
                    "gpu_peak_memory_allocated_gb": (
                        torch_module.cuda.max_memory_allocated(device) / (1024.0**3)
                    ),
                    "gpu_memory_free_gb": free_bytes / (1024.0**3),
                    "gpu_memory_total_gb": total_bytes / (1024.0**3),
                }
            )
        return snapshot

    def write(
        self,
        record: Dict[str, Any],
        batch_identity: Dict[str, Any],
        *,
        torch_module: Any,
        device: Any,
    ) -> None:
        payload = {
            "rank": self.rank,
            "local_rank": self.local_rank,
            "world_size": self.world_size,
            "host": self.hostname,
            **batch_identity,
            **{
                key: _jsonable(record.get(key))
                for key in TRACE_RECORD_KEYS
                if key in record
            },
            **self._resource_snapshot(torch_module, device),
        }
        self._handle.write(json.dumps(payload, sort_keys=True) + "\n")
        self._records_since_flush += 1
        if self._records_since_flush >= self.flush_interval:
            self._handle.flush()
            self._records_since_flush = 0

    def close(self) -> None:
        if not self._handle.closed:
            self._handle.flush()
            self._handle.close()
