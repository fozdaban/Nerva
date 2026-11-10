"""Single-sensor gait timing and wrist swing proxies, without bilateral claims."""

from typing import Any, Mapping

import numpy as np
from scipy.integrate import cumulative_trapezoid
from scipy.signal import find_peaks

from motor_pipeline.config import load_config
from motor_pipeline.io import Recording
from motor_pipeline.preprocess import prepare_imu
from motor_pipeline.quality import assess_imu


def _validate(settings: Mapping[str, Any], config: Mapping[str, Any]) -> None:
    for key in ("minimum_peak_distance_s", "minimum_gyro_rms_deg_s",
                "minimum_peak_prominence_deg_s", "event_merge_tolerance_s"):
        if not np.isfinite(settings[key]) or settings[key] <= 0:
            raise ValueError(f"{key} must be finite and positive")
    for key in ("relative_peak_prominence", "minimum_valid_window_fraction"):
        if not np.isfinite(settings[key]) or not 0 < settings[key] <= 1:
            raise ValueError(f"{key} must be in (0, 1]")
    if not np.isfinite(settings["minimum_autocorrelation"]) or not 0 <= settings["minimum_autocorrelation"] <= 1:
        raise ValueError("minimum_autocorrelation must be in [0, 1]")
    low, high = settings["stride_time_bounds_s"]
    if not 0 < low < high < config["preprocess"]["window_s"]:
        raise ValueError("Stride time bounds must be positive, ordered, and below the window length")
    if not settings["event_merge_tolerance_s"] < low / 2:
        raise ValueError("Event merge tolerance must be below half the shortest stride")
    if settings["minimum_peak_distance_s"] > low:
        raise ValueError("Peak distance must not exceed the shortest accepted stride")
    if settings["gyro_axis"] not in ("auto", "gx", "gy", "gz", "-gx", "-gy", "-gz"):
        raise ValueError("gyro_axis must be auto or a signed gyroscope axis")
    length = settings["assumed_stride_length_m"]
    if length is not None and (not np.isfinite(length) or length <= 0):
        raise ValueError("assumed_stride_length_m must be null or finite and positive")
    if not isinstance(settings["wrist_one_cycle_per_stride"], bool):
        raise ValueError("wrist_one_cycle_per_stride must be boolean")
    minimum = config["uncertainty"]["minimum_valid_windows"]
    if not isinstance(minimum, int) or minimum < 1:
        raise ValueError("minimum_valid_windows must be a positive integer")


def _projection(windows: np.ndarray, axis: str) -> np.ndarray:
    if axis != "auto":
        direction = np.zeros(3)
        direction[("gx", "gy", "gz").index(axis.removeprefix("-"))] = -1 if axis.startswith("-") else 1
        return direction
    values = windows[:, :, 3:].reshape(-1, 3)
    covariance = np.cov(values, rowvar=False)
    _, vectors = np.linalg.eigh(covariance)
    direction = vectors[:, -1]
    # Deterministic sign; without a mounting calibration, the named mid-swing
    # phase cannot be established. Either sign still gives one event per cycle.
    if direction[np.argmax(np.abs(direction))] < 0:
        direction = -direction
    return direction


def _cv(values: np.ndarray) -> float | None:
    if len(values) < 2 or np.mean(values) <= 0:
        return None
    return float(np.std(values, ddof=1) / np.mean(values))


def _window(signal: np.ndarray, fs: float, settings: Mapping[str, Any]) -> tuple[np.ndarray, np.ndarray, float] | None:
    if np.sqrt(np.mean(signal ** 2)) < settings["minimum_gyro_rms_deg_s"]:
        return None
    prominence = max(settings["minimum_peak_prominence_deg_s"], settings["relative_peak_prominence"] * np.ptp(signal))
    peaks, _ = find_peaks(signal, distance=max(1, int(np.ceil(settings["minimum_peak_distance_s"] * fs))),
                          prominence=prominence)
    intervals = np.diff(peaks) / fs
    low, high = settings["stride_time_bounds_s"]
    if not len(intervals) or np.any(intervals < low - 1e-9) or np.any(intervals > high + 1e-9):
        return None
    lag = int(round(float(np.median(intervals)) * fs))
    a, b = signal[:-lag], signal[lag:]
    a, b = a - a.mean(), b - b.mean()
    denominator = np.linalg.norm(a) * np.linalg.norm(b)
    correlation = float(np.dot(a, b) / denominator) if denominator else 0.0
    if correlation < settings["minimum_autocorrelation"]:
        return None
    amplitudes = []
    for first, last in zip(peaks[:-1], peaks[1:]):
        angle = cumulative_trapezoid(signal[first:last + 1], dx=1 / fs, initial=0)
        # Close each cycle linearly to reduce integration drift.
        angle -= np.linspace(0, angle[-1], len(angle))
        amplitudes.append(float(np.ptp(angle)))
    return peaks, np.asarray(amplitudes), correlation


def extract_gait(recording: Recording, config: Mapping[str, Any] | None = None) -> dict[str, Any]:
    """Return internal gait evidence or retest; never infer left/right asymmetry.

    Ankle events are instrumented-limb cycles, mapped to strides. Wrist cycles
    become strides only when explicitly configured, otherwise timing is reported
    as arm-swing cycles. Counts are deduplicated across overlapping windows.
    """
    config = load_config() if config is None else config
    settings = config["features"]["gait"]
    _validate(settings, config)
    # Explicit recording placement wins. The config fallback applies only when
    # metadata has no placement; canonical default metadata currently says wrist.
    placement = recording.metadata.get("placement") or settings["placement"]
    if placement not in ("ankle", "wrist"):
        return {"status": "retest", "reasons": ["Gait requires an explicitly supported ankle or wrist placement."]}
    quality = assess_imu(recording, "gait", config)
    if quality.status == "retest":
        return quality.retest_response()
    prepared = prepare_imu(recording, config, settings["filter_band_hz"])
    if not len(prepared.starts_s):
        return {"status": "retest", "reasons": ["No complete gait windows remain after quality exclusions and filter trimming."]}
    direction = _projection(prepared.bandpassed, settings["gyro_axis"])
    fs = prepared.sampling_rate_hz
    calibrated_cycles = placement == "ankle" or settings["wrist_one_cycle_per_stride"]
    rows, event_times, supported_pairs = [], [], []
    for start, values in zip(prepared.starts_s, prepared.bandpassed):
        signal = values[:, 3:] @ direction
        evidence = _window(signal, fs, settings)
        if evidence is None:
            continue
        peaks, amplitudes, correlation = evidence
        times = start + peaks / fs
        intervals = np.diff(times)
        mean = float(np.mean(intervals))
        cycle_cv = _cv(intervals)
        accel_rms = float(np.sqrt(np.mean(np.sum(values[:, :3] ** 2, axis=1))))
        features = {
            "cadence_steps_min": 120 / mean if calibrated_cycles else None,
            "mean_stride_time_s": mean if calibrated_cycles else None,
            "stride_time_cv": cycle_cv if calibrated_cycles else None,
            "number_of_strides": int(len(intervals)) if calibrated_cycles else None,
            "walking_speed_proxy_m_s": settings["assumed_stride_length_m"] / mean if calibrated_cycles and settings["assumed_stride_length_m"] is not None else None,
            "arm_swing_amplitude_deg": float(np.median(amplitudes)) if placement == "wrist" else None,
            "arm_swing_amplitude_cv": _cv(amplitudes) if placement == "wrist" else None,
            "arm_swing_cycles_min": 60 / mean if placement == "wrist" else None,
            "acceleration_rms_m_s2": accel_rms,
            "turn_count": None,
        }
        rows.append({"start_s": start, "features": features, "event_times_s": times.tolist(),
                     "cycle_autocorrelation": correlation})
        event_times.extend(times.tolist())
        supported_pairs.extend(zip(times[:-1], times[1:]))
    count = len(rows)
    coverage = count / prepared.expected_windows if prepared.expected_windows else 0.0
    if count < config["uncertainty"]["minimum_valid_windows"] or coverage < settings["minimum_valid_window_fraction"]:
        return {"status": "retest", "reasons": [
            f"Too few supported gait-cycle windows: {count} valid of {prepared.expected_windows} expected; check signal, placement and task."
        ]}
    # Assign nearby detections to one event. Anchor the tolerance to the first
    # member, rather than chaining arbitrarily distant neighbouring detections.
    clusters: list[list[float]] = []
    for event in sorted(event_times):
        if not clusters or event - clusters[-1][0] > settings["event_merge_tolerance_s"]:
            clusters.append([event])
        else:
            clusters[-1].append(event)
    unique = np.array([np.median(group) for group in clusters])
    # Count only intervals actually supported by a complete valid window. Never
    # turn a missing interval or gap between segments into an extra stride.
    pair_indices = set()
    for first, last in supported_pairs:
        left = int(np.argmin(np.abs(unique - first)))
        right = int(np.argmin(np.abs(unique - last)))
        if right == left + 1:
            pair_indices.add((left, right))
    intervals = np.array([unique[right] - unique[left] for left, right in sorted(pair_indices)])
    if not len(intervals):
        return {"status": "retest", "reasons": ["No complete gait cycles remain after event reconciliation."]}
    features, spread = {}, {}
    for name in rows[0]["features"]:
        values = [r["features"][name] for r in rows if r["features"][name] is not None]
        features[name] = float(np.median(values)) if values else None
        spread[name] = {"iqr": np.percentile(values, [25, 75]).tolist() if values else None,
                        "n_valid_windows": len(values), "quality_score": quality.score}
    if calibrated_cycles:
        mean = float(np.mean(intervals))
        features.update(cadence_steps_min=120 / mean, mean_stride_time_s=mean,
                        stride_time_cv=_cv(intervals), number_of_strides=int(len(intervals)),
                        walking_speed_proxy_m_s=settings["assumed_stride_length_m"] / mean if settings["assumed_stride_length_m"] is not None else None)
    if placement == "wrist":
        features["arm_swing_cycles_min"] = 60 / float(np.mean(intervals))
    limitations = [
        "One sensor cannot measure true left/right asymmetry.",
        "Detected cycles do not establish walking-task adherence or clinical status.",
        "Turn counting is unavailable without a validated orientation/turn method.",
        "Counts cover complete detected cycles in accepted windows; they do not extrapolate to excluded time."
    ]
    if settings["assumed_stride_length_m"] is None:
        limitations.append("Speed is withheld until an assumed or measured stride length is configured; timing alone does not measure distance.")
    else:
        limitations.append("Speed proxy assumes the configured constant stride length; it is not measured walking speed.")
    if placement == "wrist":
        limitations.append("Arm-swing angle is an integrated dominant-axis proxy, not anatomical shoulder range of motion.")
        if not calibrated_cycles:
            limitations.append("Wrist swing is not proven to equal strides; cadence, stride timing and stride count are withheld.")
    return {"status": "ok", "quality": quality.as_dict(), "features": features,
            "per_window": rows, "spread": spread, "n_valid_windows": count,
            "expected_windows": prepared.expected_windows, "placement": placement,
            "event_times_s": unique.tolist(), "cycle_intervals_s": intervals.tolist(),
            "projection_xyz": direction.tolist(), "limitations": limitations}
