"""Opt-in CUDA event timings across both inference threads.

Events are inclusive module spans, not additive percentages of wall time.
Concurrent GPU work and CPU launch gaps can affect event elapsed times.
Never enable this when reporting production FPS.
"""
from collections import defaultdict
from contextlib import contextmanager
import statistics
import threading
import torch


class StageProfiler:
    def __init__(self):
        self.events = defaultdict(list)
        self.hooks = []
        self.restores = []
        self.stacks = defaultdict(list)
        self.lock = threading.Lock()

    @contextmanager
    def span(self, name):
        start, end = (torch.cuda.Event(enable_timing=True) for _ in range(2))
        start.record()
        try:
            yield
        finally:
            end.record()
            with self.lock:
                self.events[name].append((start, end))

    def module(self, name, module):
        def before(_module, _args):
            ctx = self.span(name)
            ctx.__enter__()
            self.stacks[(name, threading.get_ident())].append(ctx)
        def after(_module, _args, _output):
            self.stacks[(name, threading.get_ident())].pop().__exit__(None, None, None)
        self.hooks += [module.register_forward_pre_hook(before),
                       module.register_forward_hook(after, always_call=True)]

    def method(self, name, owner, attribute):
        original = getattr(owner, attribute)
        def wrapped(*args, **kwargs):
            with self.span(name):
                return original(*args, **kwargs)
        self.restores.append((owner, attribute, original))
        setattr(owner, attribute, wrapped)

    def close(self):
        for h in self.hooks:
            h.remove()
        for owner, attribute, original in reversed(self.restores):
            setattr(owner, attribute, original)
        self.hooks.clear(); self.restores.clear()

    def summary(self):
        torch.cuda.synchronize()
        result = {}
        for name, events in self.events.items():
            times = [a.elapsed_time(b) for a,b in events]
            steady = times[1:] or times
            result[name] = dict(calls=len(times),total_ms=sum(times),
                                mean_ms=statistics.mean(times),
                                steady_median_ms=statistics.median(steady))
        return result
