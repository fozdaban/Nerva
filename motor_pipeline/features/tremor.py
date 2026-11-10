"""Windowed Welch tremor-band features; rule thresholds are engineering defaults."""

from typing import Any, Mapping

import numpy as np
from scipy.signal import welch

from motor_pipeline.config import load_config
from motor_pipeline.io import Recording
from motor_pipeline.preprocess import prepare_imu
from motor_pipeline.quality import assess_imu


def _band_power(f: np.ndarray, psd: np.ndarray, band: list[float]) -> float:
    """Trapezoidal integration with interpolated boundary bins (NumPy 1.24+)."""
    low, high = band
    inside = (f > low) & (f < high)
    x = np.r_[low, f[inside], high]
    y = np.r_[np.interp(low, f, psd), psd[inside], np.interp(high, f, psd)]
    return float(np.sum(np.diff(x) * (y[1:] + y[:-1]) / 2))


def _peak(f: np.ndarray, power: np.ndarray, band: list[float]) -> float:
    allowed = np.flatnonzero((f >= band[0]) & (f <= band[1]))
    index = int(allowed[np.argmax(power[allowed])])
    frequency = float(f[index])
    if 0 < index < len(power) - 1:
        a, b, c = np.log(np.maximum(power[index - 1:index + 2], np.finfo(float).tiny))
        denominator = a - 2 * b + c
        if denominator < 0:
            shift = float(np.clip(.5 * (a - c) / denominator, -.5, .5))
            frequency += shift * (f[1] - f[0])
    return float(np.clip(frequency, *band))


def _sensor_features(dynamic: np.ndarray, filtered: np.ndarray, fs: float,
                     settings: Mapping[str, Any], sensor: str) -> dict[str, float | None]:
    nperseg = min(len(dynamic), int(round(settings["welch_segment_s"] * fs)))
    f, axes_psd = welch(dynamic, fs=fs, window="hann", nperseg=nperseg,
                         noverlap=nperseg // 2, axis=0, detrend="constant", scaling="density")
    # Sum axis powers, rather than taking PSD of a rectified magnitude. For a
    # fixed orthogonal rotation this trace and the vector RMS are invariant.
    trace = axes_psd.sum(axis=1)
    band = settings["search_band_hz"]
    power = _band_power(f, trace, band)
    total = _band_power(f, trace, [0, float(f[-1])])
    ratio = float(np.clip(power / total, 0, 1)) if total > 0 else 0.0
    selected = (f >= band[0]) & (f <= band[1])
    background = (f >= max(.5, f[1])) & ~selected
    floor = float(np.median(trace[background])) if background.any() else 0.0
    spectral_peak = float(trace[selected].max())
    peak_ratio = spectral_peak / max(floor, np.finfo(float).tiny)
    present = (np.sqrt(power) >= settings["minimum_band_rms"][sensor]
               and ratio >= settings["minimum_band_ratio"]
               and peak_ratio >= settings["minimum_peak_to_background"])
    # Magnitude PSD is retained as supplemental evidence and may contain 2f.
    magnitude = np.linalg.norm(dynamic, axis=1)
    _, magnitude_psd = welch(magnitude, fs=fs, window="hann", nperseg=nperseg,
                             noverlap=nperseg // 2, detrend="constant", scaling="density")
    magnitude_total = _band_power(f, magnitude_psd, [0, float(f[-1])])
    magnitude_band = _band_power(f, magnitude_psd, band)
    return {
        f"{sensor}_dominant_freq_hz": _peak(f, trace, band) if present else None,
        f"{sensor}_power_narrow": _band_power(f, trace, settings["narrow_band_hz"]),
        f"{sensor}_power_band": power,
        f"{sensor}_band_power_ratio": ratio,
        f"{sensor}_rms": float(np.sqrt(np.mean(np.sum(filtered ** 2, axis=1)))),
        f"{sensor}_magnitude_power_band": magnitude_band,
        f"{sensor}_magnitude_band_power_ratio": float(np.clip(magnitude_band / magnitude_total, 0, 1)) if magnitude_total > 0 else 0.0,
        f"{sensor}_tremor_present": float(present),
    }


def _validate(settings: Mapping[str, Any], config: Mapping[str, Any]) -> None:
    low, high = settings["search_band_hz"]
    narrow_low, narrow_high = settings["narrow_band_hz"]
    filter_low, filter_high = settings["filter_band_hz"]
    if not 0 < filter_low <= low <= narrow_low < narrow_high <= high <= filter_high:
        raise ValueError("Tremor bands must be nested within the filter band")
    if settings["primary_sensor"] not in ("accel", "gyro"):
        raise ValueError("primary_sensor must be accel or gyro")
    for key in ("minimum_band_ratio", "minimum_valid_window_fraction"):
        if not np.isfinite(settings[key]) or not 0 < settings[key] <= 1:
            raise ValueError(f"{key} must be in (0, 1]")
    for value in (settings["minimum_peak_to_background"], *settings["minimum_band_rms"].values()):
        if not np.isfinite(value) or value <= 0:
            raise ValueError("Peak and amplitude thresholds must be finite and positive")
    seconds = settings["welch_segment_s"]
    if not np.isfinite(seconds) or not 1 <= seconds <= config["preprocess"]["window_s"]:
        raise ValueError("welch_segment_s must be between 1 s and window_s")
    minimum = config["uncertainty"]["minimum_valid_windows"]
    if not isinstance(minimum, int) or minimum < 1:
        raise ValueError("minimum_valid_windows must be a positive integer")


def extract_tremor(recording: Recording, config: Mapping[str, Any] | None = None) -> dict[str, Any]:
    """Return internal named features and per-window values, or minimal retest.

    This is a feature-stage result, not the proposed mobile API envelope. Median
    values and IQRs describe overlapping windows; they are not 95% confidence
    intervals. Frequency is withheld when no window meets the signal rule.
    """
    config = load_config() if config is None else config
    settings = config["features"]["tremor"]
    _validate(settings, config)
    quality = assess_imu(recording, "tremor", config)
    if quality.status == "retest":
        return quality.retest_response()
    prepared = prepare_imu(recording, config, settings["filter_band_hz"])
    count = len(prepared.starts_s)
    coverage = count / prepared.expected_windows if prepared.expected_windows else 0.0
    if count < config["uncertainty"]["minimum_valid_windows"] or coverage < settings["minimum_valid_window_fraction"]:
        return {"status": "retest", "reasons": [
            f"Too few complete tremor windows after excluding gaps, missing values, clipping and filter edges: {count} valid of {prepared.expected_windows} expected."
        ]}
    per_window = []
    for start, dynamic, filtered in zip(prepared.starts_s, prepared.dynamic, prepared.bandpassed):
        features = {}
        for sensor, indices in (("accel", slice(0, 3)), ("gyro", slice(3, 6))):
            features.update(_sensor_features(dynamic[:, indices], filtered[:, indices],
                                             prepared.sampling_rate_hz, settings, sensor))
        per_window.append({"start_s": start, "features": features})
    aggregate, spread = {}, {}
    for name in per_window[0]["features"]:
        values = [row["features"][name] for row in per_window if row["features"][name] is not None]
        aggregate[name] = float(np.median(values)) if values else None
        spread[name] = {"iqr": [float(v) for v in np.percentile(values, [25, 75])] if values else None,
                        "n_valid_windows": len(values), "quality_score": quality.score}
    # Fractions are means of 0/1 window indicators, never medians.
    for sensor in ("accel", "gyro"):
        present = [row["features"][f"{sensor}_tremor_present"] for row in per_window]
        aggregate[f"{sensor}_tremor_present"] = float(np.mean(present))
        frequencies = [row["features"][f"{sensor}_dominant_freq_hz"] for row in per_window
                       if row["features"][f"{sensor}_dominant_freq_hz"] is not None]
        aggregate[f"{sensor}_peak_frequency_sd_hz"] = float(np.std(frequencies, ddof=1)) if len(frequencies) >= 2 else None
        spread[f"{sensor}_peak_frequency_sd_hz"] = {"iqr": None, "n_valid_windows": len(frequencies), "quality_score": quality.score}
    primary = settings["primary_sensor"]
    aggregate["dominant_freq_hz"] = aggregate[f"{primary}_dominant_freq_hz"]
    aggregate["tremor_present_fraction"] = aggregate[f"{primary}_tremor_present"]
    spread["dominant_freq_hz"] = dict(spread[f"{primary}_dominant_freq_hz"])
    spread["tremor_present_fraction"] = dict(spread[f"{primary}_tremor_present"])
    return {"status": "ok", "quality": quality.as_dict(), "features": aggregate,
            "per_window": per_window, "spread": spread,
            "n_valid_windows": count, "expected_windows": prepared.expected_windows}
