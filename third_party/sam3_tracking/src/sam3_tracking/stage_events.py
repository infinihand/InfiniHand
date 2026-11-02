"""Optional JSONL stage timing for standalone runs."""

import contextlib
import json
import os
import time


def stage_events_enabled():
    return os.environ.get("HAND_SAM3_STAGE_EVENTS", "0") == "1"


def phase19_stage_event(stage, phase, **kwargs):
    if not stage_events_enabled():
        return
    record = {
        "stage": stage,
        "phase": phase,
        "time_unix": time.time(),
        "pid": os.getpid(),
        "slurm_rank": os.environ.get("SLURM_PROCID", ""),
        **kwargs,
    }
    path = os.environ["HAND_SAM3_STAGE_EVENTS_PATH"]
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, sort_keys=True) + "\n")


@contextlib.contextmanager
def stage_event_scope(stage, **kwargs):
    phase19_stage_event(stage, "start", **kwargs)
    try:
        yield
    finally:
        phase19_stage_event(stage, "end", **kwargs)
