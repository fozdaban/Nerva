"""Canonical 2D video evidence; keep missing frames instead of zero filling."""

from dataclasses import dataclass, field
from typing import Any

import numpy as np


@dataclass
class LandmarkRecording:
    """Normalized MediaPipe coordinates, timestamps, dimensions and visibility.

    ``points`` is (frames, 21 or 33 landmarks, xyz); NaN means not tracked.
    ``confidence`` is per-landmark visibility/presence when actually available,
    never a handedness classification score. Hands normally have None here.
    """

    t: np.ndarray
    points: np.ndarray
    kind: str
    width: int
    height: int
    confidence: np.ndarray | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    diagnostics: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.t = np.asarray(self.t, dtype=float).copy()
        self.points = np.asarray(self.points, dtype=float).copy()
        if self.kind not in ("hand", "pose"):
            raise ValueError("kind must be hand or pose")
        count = 21 if self.kind == "hand" else 33
        if self.t.ndim != 1 or self.points.shape != (len(self.t), count, 3):
            raise ValueError(f"points must have shape (frames, {count}, 3)")
        if self.width <= 0 or self.height <= 0:
            raise ValueError("Frame dimensions must be positive")
        if self.confidence is not None:
            self.confidence = np.asarray(self.confidence, dtype=float).copy()
            if self.confidence.shape != self.points.shape[:2]:
                raise ValueError("confidence must have shape (frames, landmarks)")

    def image_xy(self) -> np.ndarray:
        """2D coordinates in frame-height units, correcting width/height scaling."""
        return self.points[:, :, :2] * np.array([self.width / self.height, 1.0])
