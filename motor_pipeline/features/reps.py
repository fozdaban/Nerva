"""Complete exercise-cycle counting and motion proxies from one IMU."""

from copy import deepcopy
from typing import Any, Mapping

import numpy as np
from scipy.integrate import cumulative_trapezoid
from scipy.signal import find_peaks

from motor_pipeline.config import load_config
from motor_pipeline.io import Recording
from motor_pipeline.preprocess import prepare_imu
from motor_pipeline.quality import assess_imu


def _validate(settings: Mapping[str, Any], config: Mapping[str, Any]) -> None:
    sensor = settings["primary_sensor"]
    if sensor not in ("accel", "gyro"):
        raise ValueError("primary_sensor must be accel or gyro")
    axes = ("ax", "ay", "az") if sensor == "accel" else ("gx", "gy", "gz")
    if settings["axis"] not in ("auto", *axes, *(f"-{axis}" for axis in axes)):
        raise ValueError("axis must be auto or a signed primary-sensor axis")
    if not settings["accepted_placements"] or not all(isinstance(v, str) for v in settings["accepted_placements"]):
        raise ValueError("accepted_placements must contain placement names")
    for key in ("minimum_peak_distance_s", "gravity_highpass_hz"):
        if not np.isfinite(settings[key]) or settings[key] <= 0:
            raise ValueError(f"{key} must be finite and positive")
    if settings["gravity_highpass_hz"] >= settings["filter_band_hz"][0]:
        raise ValueError("gravity_highpass_hz must be below the repetition filter band")
    for key in ("relative_peak_prominence", "minimum_valid_window_fraction"):
        if not np.isfinite(settings[key]) or not 0 < settings[key] <= 1:
            raise ValueError(f"{key} must be in (0, 1]")
    for key in ("minimum_signal_rms", "minimum_peak_prominence"):
        for value in settings[key].values():
            if not np.isfinite(value) or value <= 0:
                raise ValueError(f"{key} thresholds must be finite and positive")
    low, high = settings["rep_duration_bounds_s"]
    if not np.isfinite(low) or not np.isfinite(high) or not 0 < low < high:
        raise ValueError("rep_duration_bounds_s must be finite, positive and ordered")
    if settings["minimum_peak_distance_s"] > low:
        raise ValueError("Peak separation must not exceed the shortest accepted repetition")
    minimum = config["uncertainty"]["minimum_valid_windows"]
    if not isinstance(minimum, int) or minimum < 1:
        raise ValueError("minimum_valid_windows must be a positive integer")
    if not isinstance(settings["trend_boundary_cycles"], int) or settings["trend_boundary_cycles"] < 1:
        raise ValueError("trend_boundary_cycles must exclude at least one cycle at each boundary")
    if not isinstance(settings["minimum_trend_cycles"], int) or settings["minimum_trend_cycles"] < 3:
        raise ValueError("minimum_trend_cycles must be an integer of at least 3 interior cycles")


def _direction(windows: np.ndarray, sensor: str, axis: str) -> np.ndarray:
    axes = ("ax", "ay", "az") if sensor == "accel" else ("gx", "gy", "gz")
    if axis != "auto":
        direction = np.zeros(3)
        direction[axes.index(axis.removeprefix("-"))] = -1 if axis.startswith("-") else 1
        return direction
    _, vectors = np.linalg.eigh(np.cov(windows.reshape(-1, 3), rowvar=False))
    direction = vectors[:, -1]
    if direction[np.argmax(np.abs(direction))] < 0:
        direction = -direction
    return direction


def _cv(values: np.ndarray) -> float | None:
    return float(np.std(values, ddof=1) / np.mean(values)) if len(values) >= 2 and np.mean(values) > 0 else None


def _metrics(durations: np.ndarray, amplitudes: np.ndarray, centers: np.ndarray,
             fit_trend: bool, minimum_trend_cycles: int, boundary_cycles: int) -> dict[str, int | float | None]:
    slope, slowing = None, None
    interior = slice(boundary_cycles, -boundary_cycles if boundary_cycles else None)
    trend_durations, trend_centers = durations[interior], centers[interior]
    if fit_trend and len(trend_durations) >= minimum_trend_cycles and trend_centers[-1] > trend_centers[0]:
        gradient, first = np.polyfit(trend_centers - trend_centers[0], trend_durations, 1)
        slope = float(gradient)
        slowing = float(gradient * (trend_centers[-1] - trend_centers[0]) / first) if first > 0 else None
    return {
        "rep_count": int(len(durations)),
        "mean_rep_duration_s": float(np.mean(durations)) if len(durations) else None,
        "rep_duration_cv": _cv(durations),
        "range_of_motion_proxy": float(np.median(amplitudes)) if len(amplitudes) else None,
        "range_of_motion_cv": _cv(amplitudes),
        "duration_slope_s_per_s": slope,
        "slowing_fraction": slowing,
    }


def extract_reps(recording: Recording, config: Mapping[str, Any] | None = None) -> dict[str, Any]:
    """Measure complete peak-to-peak cycles, not observed peaks or partial reps.

    Gyro mode integrates projected angular velocity per complete cycle and
    returns an angle proxy. Acceleration mode reports projected acceleration
    excursion and never labels that signal as anatomical movement distance.
    """
    config = load_config() if config is None else config
    settings = config["features"]["reps"]
    _validate(settings, config)
    if recording.metadata.get("placement") not in settings["accepted_placements"]:
        return {"status": "retest", "reasons": ["Recording placement is not configured for this exercise."]}
    quality = assess_imu(recording, "reps", config)
    if quality.status == "retest":
        return quality.retest_response()
    preprocessing_config = deepcopy(config)
    preprocessing_config["preprocess"]["gravity_highpass_hz"] = settings["gravity_highpass_hz"]
    prepared = prepare_imu(recording, preprocessing_config, settings["filter_band_hz"])
    if not len(prepared.starts_s):
        return {"status": "retest", "reasons": ["No complete exercise windows remain after exclusions."]}
    sensor = settings["primary_sensor"]
    selected = slice(0, 3) if sensor == "accel" else slice(3, 6)
    direction = _direction(prepared.bandpassed[:, :, selected], sensor, settings["axis"])
    projected = prepared.bandpassed[:, :, selected] @ direction
    fs = prepared.sampling_rate_hz
    accepted = np.sqrt(np.mean(projected ** 2, axis=1)) >= settings["minimum_signal_rms"][sensor]
    count = int(accepted.sum())
    coverage = count / prepared.expected_windows if prepared.expected_windows else 0
    if count < config["uncertainty"]["minimum_valid_windows"] or coverage < settings["minimum_valid_window_fraction"]:
        return {"status": "retest", "reasons": [f"Too few exercise windows with sufficient motion: {count} of {prepared.expected_windows} expected."]}
    # Stitch only the union of accepted windows. Uncovered holes split separate
    # analysis runs even if all samples originally belonged to one source segment.
    # Shared samples come from the same filtered source; overlap is not a new rep.
    events, cycles, analysis_runs = [], [], []
    for segment_id in sorted(set(prepared.segment_ids)):
        indices = [i for i, sid in enumerate(prepared.segment_ids) if sid == segment_id and accepted[i]]
        groups: list[list[int]] = []
        for index in indices:
            if not groups or prepared.starts_s[index] > prepared.starts_s[groups[-1][-1]] + projected.shape[1] / fs + 1e-9:
                groups.append([index])
            else:
                groups[-1].append(index)
        for group in groups:
            start = prepared.starts_s[group[0]]
            last_offset = round((prepared.starts_s[group[-1]] - start) * fs)
            signal = np.full(last_offset + projected.shape[1], np.nan)
            for index in group:
                offset = round((prepared.starts_s[index] - start) * fs)
                signal[offset:offset + projected.shape[1]] = projected[index]
            if not np.isfinite(signal).all():
                return {"status": "retest", "reasons": ["Exercise samples cannot be joined without an unsupported interval."]}
            run_id = len(analysis_runs)
            analysis_runs.append({"analysis_run_id": run_id, "segment_id": segment_id,
                                  "start_s": float(start), "end_s": float(start + len(signal) / fs)})
            prominence = max(settings["minimum_peak_prominence"][sensor], settings["relative_peak_prominence"] * np.ptp(signal))
            peaks, _ = find_peaks(signal, distance=max(1, int(np.ceil(settings["minimum_peak_distance_s"] * fs))), prominence=prominence)
            times = start + peaks / fs
            events.extend({"time_s": float(t), "segment_id": segment_id, "analysis_run_id": run_id} for t in times)
            for left, right, first, last in zip(peaks[:-1], peaks[1:], times[:-1], times[1:]):
                duration = float(last - first)
                if not settings["rep_duration_bounds_s"][0] - 1e-9 <= duration <= settings["rep_duration_bounds_s"][1] + 1e-9:
                    return {"status": "retest", "reasons": ["Detected exercise cycle duration is outside the configured bounds; check task and peak settings."]}
                values = signal[left:right + 1]
                if sensor == "gyro":
                    angle = cumulative_trapezoid(values, dx=1 / fs, initial=0)
                    angle -= np.linspace(0, angle[-1], len(angle))
                    amplitude = float(np.ptp(angle))
                else:
                    amplitude = float(np.ptp(values))
                cycles.append({"start_s": float(first), "end_s": float(last),
                               "duration_s": duration, "range_of_motion_proxy": amplitude,
                               "segment_id": segment_id, "analysis_run_id": run_id})
    if len(cycles) < 2:
        return {"status": "retest", "reasons": ["At least two complete exercise cycles are required."]}
    def metrics_for(selected_cycles: list[dict[str, Any]], allow_trend: bool, boundary_cycles: int = 0) -> dict[str, Any]:
        durations = np.array([c["duration_s"] for c in selected_cycles])
        amplitudes = np.array([c["range_of_motion_proxy"] for c in selected_cycles])
        centers = np.array([(c["start_s"] + c["end_s"]) / 2 for c in selected_cycles])
        return _metrics(durations, amplitudes, centers, allow_trend, settings["minimum_trend_cycles"], boundary_cycles)
    continuous = len({c["analysis_run_id"] for c in cycles}) == 1
    features = metrics_for(cycles, continuous, settings["trend_boundary_cycles"])
    rows = []
    for i, start in enumerate(prepared.starts_s):
        if not accepted[i]:
            continue
        end = start + projected.shape[1] / fs
        subset = [c for c in cycles if c["segment_id"] == prepared.segment_ids[i] and c["start_s"] >= start - 1e-9 and c["end_s"] < end + 1e-9]
        rows.append({"start_s": start, "end_s": end, "segment_id": prepared.segment_ids[i], "features": metrics_for(subset, True)})
    spread = {}
    for name in features:
        values = [r["features"][name] for r in rows if r["features"][name] is not None]
        spread[name] = {"iqr": np.percentile(values, [25, 75]).tolist() if values else None,
                        "n_valid_windows": len(values), "quality_score": quality.score}
    limitations = [
        "One sensor measures periodic motion, not verified exercise form or task adherence.",
        "Counts represent complete same-phase peak-to-peak cycles in observed prepared segments; partial cycles are excluded.",
        "Range of motion is a dominant-axis signal proxy, not anatomical joint range or displacement.",
        "Window IQRs describe overlapping window estimates, not confidence intervals for global counts or trends."
    ]
    if not continuous:
        limitations.append("Slowing trend is withheld across discontinuous analysis runs; missing or unsupported data is not a slow repetition.")
    if features["duration_slope_s_per_s"] is None:
        limitations.append("A global slowing trend requires sufficient interior complete cycles after excluding boundary cycles from the fit.")
    return {"status": "ok", "quality": quality.as_dict(), "features": features,
            "per_window": rows, "spread": spread, "n_valid_windows": count,
            "expected_windows": prepared.expected_windows, "events": events, "cycles": cycles,
            "analysis_runs": analysis_runs,
            "primary_sensor": sensor, "projection_xyz": direction.tolist(),
            "range_of_motion_unit": "deg" if sensor == "gyro" else "m/s2",
            "limitations": limitations}
