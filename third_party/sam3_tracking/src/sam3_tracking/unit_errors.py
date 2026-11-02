"""Typed recoverable unit-level errors for Ego4D SAM3 HaWoR."""


class RecoverableUnitError(Exception):
    """An enumerated per-unit failure that should skip the unit, not crash a video."""

    def __init__(self, reason: str, detail: dict | None = None):
        self.reason = str(reason)
        self.detail = dict(detail or {})
        super().__init__(self.reason)

    def __str__(self):
        return f"{self.reason}: {self.detail}" if self.detail else self.reason
