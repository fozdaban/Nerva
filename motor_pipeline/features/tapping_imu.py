"""Wrist motion tap proxies with deduplicated events and conservative gap handling."""

from typing import Any, Mapping

import numpy as np
from scipy.signal import find_peaks

from motor_pipeline.config import load_config
from motor_pipeline.io import Recording
from motor_pipeline.preprocess import prepare_imu
from motor_pipeline.quality import assess_imu


def _validate(settings: Mapping[str, Any], config: Mapping[str, Any]) -> None:
    if settings["primary_sensor"] not in ("accel", "gyro"):
        raise ValueError("primary_sensor must be accel or gyro")
    axes = ("ax", "ay", "az") if settings["primary_sensor"] == "accel" else ("gx", "gy", "gz")
    if settings["axis"] not in ("auto", *axes, *(f"-{axis}" for axis in axes)):
        raise ValueError("axis must be auto or a signed axis of the primary sensor")
    for key in ("minimum_peak_distance_s", "event_merge_tolerance_s"):
        if not np.isfinite(settings[key]) or settings[key] <= 0:
            raise ValueError(f"{key} must be finite and positive")
    if settings["event_merge_tolerance_s"] >= settings["minimum_peak_distance_s"] / 2:
        raise ValueError("Merge tolerance must be below half the minimum peak distance")
    for key in ("relative_peak_prominence", "minimum_valid_window_fraction"):
        if not np.isfinite(settings[key]) or not 0 < settings[key] <= 1:
            raise ValueError(f"{key} must be in (0, 1]")
    for name in ("minimum_peak_prominence", "minimum_signal_rms"):
        for sensor in ("accel", "gyro"):
            value = settings[name][sensor]
            if not np.isfinite(value) or value <= 0:
                raise ValueError(f"{name}.{sensor} must be finite and positive")
    if not np.isfinite(settings["hesitation_median_multiplier"]) or settings["hesitation_median_multiplier"] <= 1:
        raise ValueError("hesitation_median_multiplier must exceed 1")
    if not isinstance(settings["minimum_peaks_per_window"], int) or settings["minimum_peaks_per_window"] < 3:
        raise ValueError("minimum_peaks_per_window must be an integer of at least 3")
    minimum = config["uncertainty"]["minimum_valid_windows"]
    if not isinstance(minimum, int) or minimum < 1:
        raise ValueError("minimum_valid_windows must be a positive integer")


def _direction(values: np.ndarray, axis: str, sensor: str) -> np.ndarray:
    axes = ("ax", "ay", "az") if sensor == "accel" else ("gx", "gy", "gz")
    if axis != "auto":
        direction = np.zeros(3)
        direction[axes.index(axis.removeprefix("-"))] = -1 if axis.startswith("-") else 1
        return direction
    _, vectors = np.linalg.eigh(np.cov(values, rowvar=False))
    direction = vectors[:, -1]
    projected = values @ direction
    # Positive sparse pulses have a positive third central moment. This makes
    # polarity robust to a fixed mounting rotation for the pulse-like fixture.
    centered = projected - projected.mean()
    if np.mean(centered ** 3) < 0:
        direction = -direction
    return direction


def _amplitude_trend(times: np.ndarray, amplitudes: np.ndarray) -> tuple[float | None, float | None]:
    if len(times) < 3 or times[-1] <= times[0]:
        return None, None
    slope, intercept = np.polyfit(times - times[0], amplitudes, 1)
    decrement = -slope * (times[-1] - times[0]) / intercept if intercept > 0 else None
    return float(slope), float(decrement) if decrement is not None else None


def _metrics(times: np.ndarray, amplitudes: np.ndarray, intervals: np.ndarray,
             settings: Mapping[str, Any], fit_trend: bool = True) -> dict[str, float | int | None]:
    mean = float(intervals.mean()) if len(intervals) else None
    cv = float(np.std(intervals, ddof=1) / mean) if len(intervals) >= 2 and mean else None
    slope, decrement = _amplitude_trend(times, amplitudes) if fit_trend else (None, None)
    return {
        "tap_count": int(len(times)), "tap_rate_hz": 1 / mean if mean else None,
        "inter_tap_interval_cv": cv, "median_tap_amplitude": float(np.median(amplitudes)),
        "amplitude_slope_per_s": slope, "amplitude_decrement_fraction": decrement,
        "hesitation_count": int(np.sum(intervals > settings["hesitation_median_multiplier"] * np.median(intervals))) if len(intervals) else None,
    }


def extract_tapping_imu(recording: Recording, config: Mapping[str, Any] | None = None) -> dict[str, Any]:
    """Return internal event/window evidence; wrist IMU does not measure fingers.

    Long intervals are measured only within an uninterrupted prepared segment.
    Gaps never become hesitations. Overlap never becomes duplicate taps.
    """
    config = load_config() if config is None else config
    settings = config["features"]["tapping_imu"]
    _validate(settings, config)
    if recording.metadata.get("placement") != "wrist":
        return {"status": "retest", "reasons": ["IMU tapping requires wrist placement metadata."]}
    quality = assess_imu(recording, "tapping_imu", config)
    if quality.status == "retest":
        return quality.retest_response()
    prepared = prepare_imu(recording, config, settings["filter_band_hz"])
    if not len(prepared.starts_s):
        return {"status": "retest", "reasons": ["No complete tapping windows remain after quality exclusions."]}
    sensor = settings["primary_sensor"]
    selected = slice(0, 3) if sensor == "accel" else slice(3, 6)
    direction = _direction(prepared.bandpassed[:, :, selected].reshape(-1, 3), settings["axis"], sensor)
    rows, detections = [], []
    fs = prepared.sampling_rate_hz
    for start, values, segment_id in zip(prepared.starts_s, prepared.bandpassed, prepared.segment_ids):
        signal = values[:, selected] @ direction
        if np.sqrt(np.mean(signal ** 2)) < settings["minimum_signal_rms"][sensor]:
            continue
        prominence = max(settings["minimum_peak_prominence"][sensor], settings["relative_peak_prominence"] * np.ptp(signal))
        peaks, properties = find_peaks(signal, distance=max(1, int(np.ceil(settings["minimum_peak_distance_s"] * fs))),
                                       prominence=prominence)
        # Retain detections even in sparse windows so a genuine pause between
        # supported tapping runs can be measured without demanding regularity.
        times = start + peaks / fs
        amplitudes = properties["prominences"]
        detections.extend((segment_id, float(t), float(a)) for t, a in zip(times, amplitudes))
        if len(peaks) < settings["minimum_peaks_per_window"]:
            continue
        rows.append({"start_s": start, "segment_id": segment_id,
                     "features": _metrics(times, amplitudes, np.diff(times), settings),
                     "event_times_s": times.tolist()})
    count = len(rows)
    coverage = count / prepared.expected_windows if prepared.expected_windows else 0.0
    if count < config["uncertainty"]["minimum_valid_windows"] or coverage < settings["minimum_valid_window_fraction"]:
        return {"status": "retest", "reasons": [f"Too few supported tapping windows: {count} valid of {prepared.expected_windows} expected."]}
    clusters: list[list[tuple[int, float, float]]] = []
    for detection in sorted(detections):
        if (not clusters or detection[0] != clusters[-1][0][0]
                or detection[1] - clusters[-1][0][1] > settings["event_merge_tolerance_s"]):
            clusters.append([detection])
        else:
            clusters[-1].append(detection)
    times = np.array([np.median([event[1] for event in group]) for group in clusters])
    amplitudes = np.array([np.median([event[2] for event in group]) for group in clusters])
    segments = np.array([group[0][0] for group in clusters])
    intervals = np.diff(times)[np.diff(segments) == 0]
    if len(intervals) < 2:
        return {"status": "retest", "reasons": ["Insufficient within-segment inter-tap intervals."]}
    features = _metrics(times, amplitudes, intervals, settings, fit_trend=len(set(segments)) == 1)
    spread = {}
    for name in features:
        values = [row["features"][name] for row in rows if row["features"][name] is not None]
        spread[name] = {"iqr": np.percentile(values, [25, 75]).tolist() if values else None,
                        "n_valid_windows": len(values), "quality_score": quality.score}
    limitations = [
        "Wrist motion peaks are proxies for taps, not direct measurements of finger opening or finger contact.",
        "Counts and timing cover prepared time only; excluded gaps are not hesitations.",
        "A long interval can reflect a pause or an undetected low-amplitude tap.",
        "Amplitude is filtered peak prominence, not finger excursion; slope is affected by mounting and filtering.",
        "Window IQRs are descriptive spread, not confidence intervals for global counts or slopes."
    ]
    if len(set(segments)) > 1:
        limitations.append("Global amplitude trend is withheld across discontinuous segments; within-window trends remain available.")
    return {"status": "ok", "quality": quality.as_dict(), "features": features,
            "per_window": rows, "spread": spread, "n_valid_windows": count,
            "expected_windows": prepared.expected_windows,
            "event_times_s": times.tolist(), "event_amplitudes": amplitudes.tolist(),
            "event_segment_ids": segments.tolist(), "inter_tap_intervals_s": intervals.tolist(),
            "projection_xyz": direction.tolist(), "primary_sensor": sensor,
            "amplitude_unit": "m/s2" if sensor == "accel" else "deg/s", "limitations": limitations}
