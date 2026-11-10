"""Complete landmark windows: never bridge a missed frame or timestamp gap."""

from typing import Any, Mapping

import numpy as np


def landmark_windows(t: np.ndarray, valid: np.ndarray, config: Mapping[str, Any]) -> tuple[list[tuple[int, np.ndarray]], int, float]:
    p = config["preprocess"]
    if not 2 <= p["window_s"] <= 5 or not 0 <= p["overlap_fraction"] < 1:
        raise ValueError("Video windows require 2–5 seconds and overlap in [0,1)")
    fs = float(1 / np.median(np.diff(t)))
    width = int(round(p["window_s"] * fs))
    hop = max(1, int(round(width * (1 - p["overlap_fraction"]))))
    duration = t[-1] - t[0] + 1 / fs
    expected = max(0, 1 + (int(round(duration * fs)) - width) // hop)
    indices = np.flatnonzero(valid)
    cuts = np.flatnonzero((np.diff(indices) > 1) | (np.diff(t[indices]) > 1.5 / fs + 1e-9)) + 1
    windows = []
    for segment_id, segment in enumerate(np.split(indices, cuts)):
        for offset in range(0, len(segment) - width + 1, hop):
            windows.append((segment_id, segment[offset:offset + width]))
    return windows, expected, fs


def feature_spread(rows: list[dict[str, Any]], names: list[str], score: float) -> dict[str, Any]:
    result = {}
    for name in names:
        values = [row["features"][name] for row in rows if row["features"][name] is not None]
        result[name] = {"iqr": np.percentile(values, [25, 75]).tolist() if values else None,
                        "n_valid_windows": len(values), "quality_score": score}
    return result


def minimum_windows(config: Mapping[str, Any]) -> int:
    value = config["uncertainty"]["minimum_valid_windows"]
    if not isinstance(value, int) or value < 1:
        raise ValueError("minimum_valid_windows must be a positive integer")
    return value
