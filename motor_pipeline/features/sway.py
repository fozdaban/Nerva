"""Acceleration-domain quiet-standing proxies, never displacement or COP."""

from fractions import Fraction
from typing import Any, Mapping

import numpy as np
from scipy.signal import butter, resample_poly, sosfiltfilt, welch
from scipy.stats import chi2

from motor_pipeline.config import load_config
from motor_pipeline.io import Recording
from motor_pipeline.quality import assess_imu


def _validate(config: Mapping[str, Any], native_fs: float) -> None:
    settings, p = config["features"]["sway"], config["preprocess"]
    axes = settings["horizontal_axes"]
    if len(axes) != 2 or len(set(axes)) != 2 or not set(axes) <= {"ax", "ay", "az"}:
        raise ValueError("horizontal_axes must be two distinct acceleration axes")
    fs = p["resample_hz"]
    if not np.isfinite(fs) or fs <= 0 or not np.isfinite(native_fs) or native_fs <= 0:
        raise ValueError("Source and target sampling rates must be finite and positive")
    low, high = settings["filter_band_hz"]
    if not 0 < low < high < min(native_fs, fs) / 2:
        raise ValueError("Sway band must be positive and below source/target Nyquist")
    if not isinstance(p["filter_order"], int) or not 1 <= p["filter_order"] <= 8:
        raise ValueError("filter_order must be an integer from 1 to 8")
    if not np.isfinite(p["window_s"]) or not 2 <= p["window_s"] <= 5:
        raise ValueError("window_s must be between 2 and 5 seconds")
    if not np.isfinite(p["overlap_fraction"]) or not 0 <= p["overlap_fraction"] < 1:
        raise ValueError("overlap_fraction must be in [0, 1)")
    for key, value in (("edge_trim_s", settings["edge_trim_s"]),
                       ("maximum_interpolation_gap_s", p["maximum_interpolation_gap_s"])):
        if not np.isfinite(value) or value <= 0:
            raise ValueError(f"{key} must be finite and positive")
    fraction = settings["minimum_valid_window_fraction"]
    if not np.isfinite(fraction) or not 0 < fraction <= 1:
        raise ValueError("minimum_valid_window_fraction must be in (0, 1]")
    if not isinstance(settings["require_horizontal_alignment_confirmation"], bool):
        raise ValueError("require_horizontal_alignment_confirmation must be boolean")
    if not settings["supported_placements"] or not all(isinstance(x, str) for x in settings["supported_placements"]):
        raise ValueError("supported_placements must contain placement names")
    minimum = config["uncertainty"]["minimum_valid_windows"]
    if not isinstance(minimum, int) or minimum < 1:
        raise ValueError("minimum_valid_windows must be a positive integer")


def _windows(recording: Recording, config: Mapping[str, Any]) -> tuple[list[tuple[float, np.ndarray, int, int]], int]:
    """Prepare acceleration alone without substituting missing gyro signals."""
    settings, p = config["features"]["sway"], config["preprocess"]
    native_fs, fs = float(recording.metadata["sampling_rate_hz"]), float(p["resample_hz"])
    t = recording.data.t.to_numpy(dtype=float)
    acceleration = recording.data[["ax", "ay", "az"]].to_numpy(dtype=float)
    full_scale = config["input"]["acceleration_range_g"] * config["input"]["standard_gravity_m_s2"]
    valid = np.isfinite(acceleration).all(axis=1) & (np.abs(acceleration) < full_scale * config["quality"]["clipping_range_fraction"]).all(axis=1)
    indices = np.flatnonzero(valid)
    max_gap = max(p["maximum_interpolation_gap_s"], 1.5 / native_fs)
    cuts = np.flatnonzero((np.diff(indices) > 1) | (np.diff(t[indices]) > max_gap + 1e-9)) + 1
    segments = np.split(indices, cuts)
    width = int(round(p["window_s"] * fs))
    hop = max(1, int(round(width * (1 - p["overlap_fraction"]))))
    trim = int(np.ceil(settings["edge_trim_s"] * fs))
    total = int(round((t[-1] - t[0] + 1 / native_fs) * fs))
    expected = max(0, 1 + (total - 2 * trim - width) // hop)
    selected = [["ax", "ay", "az"].index(axis) for axis in settings["horizontal_axes"]]
    sos = butter(p["filter_order"], settings["filter_band_hz"], btype="bandpass", fs=fs, output="sos")
    ratio = (Fraction(str(fs)) / Fraction(str(native_fs))).limit_denominator(1000)
    rows = []
    for segment_id, segment in enumerate(segments):
        if len(segment) < 2:
            continue
        start, end = float(t[segment[0]]), float(t[segment[-1]])
        if end - start + 1 / native_fs < p["window_s"] + 2 * settings["edge_trim_s"] - 1e-9:
            continue
        native_t = start + np.arange(int(np.floor((end - start) * native_fs + 1e-7)) + 1) / native_fs
        regular = np.column_stack([np.interp(native_t, t[segment], acceleration[segment, axis]) for axis in selected])
        regular -= regular.mean(axis=0)
        resampled = resample_poly(regular, ratio.numerator, ratio.denominator, axis=0)
        limit = int(np.floor((end - start) * fs + 1e-7)) + 1
        resampled = resampled[:limit]
        if len(resampled) < width + 2 * trim:
            continue
        filtered = sosfiltfilt(sos, resampled, axis=0)
        for offset in range(trim, len(filtered) - trim - width + 1, hop):
            rows.append((float(start + offset / fs), filtered[offset:offset + width], segment_id, offset))
    return rows, expected


def _metrics(values: np.ndarray, fs: float) -> dict[str, float | None]:
    """Area is a Gaussian covariance ellipse, not a statistical CI on sway."""
    centered = values - values.mean(axis=0)
    rms = float(np.sqrt(np.mean(np.sum(centered ** 2, axis=1))))
    path = float(np.linalg.norm(np.diff(centered, axis=0), axis=1).sum())
    covariance = np.cov(centered, rowvar=False)
    determinant = max(0.0, float(np.linalg.det(covariance)))
    ellipse = float(np.pi * chi2.ppf(.95, df=2) * np.sqrt(determinant))
    f, psd = welch(centered, fs=fs, window="hann", nperseg=len(centered),
                   noverlap=len(centered) // 2, axis=0, scaling="density", detrend="constant")
    power = psd.sum(axis=1)
    power[0] = 0
    total = float(power.sum())
    mean_frequency = float(np.dot(f, power) / total) if total > np.finfo(float).tiny else None
    return {"horizontal_acceleration_rms_m_s2": rms,
            "acceleration_path_length_proxy_m_s2": path,
            "mean_frequency_hz": mean_frequency,
            "acceleration_ellipse_area_95_m2_s4": ellipse}


def extract_sway(recording: Recording, config: Mapping[str, Any] | None = None) -> dict[str, Any]:
    """Measure horizontal acceleration variation during quiet standing.

    Mounting must identify horizontal axes. Missing sensor values and time gaps
    split independently filtered segments; no displacement integration is used.
    """
    config = load_config() if config is None else config
    settings = config["features"]["sway"]
    # Rate validation is completed by quality below; validate configuration with
    # its nominal value even when caller-created metadata itself is corrupt.
    _validate(config, float(config["input"]["sampling_rate_hz"]))
    placement = recording.metadata.get("placement") or settings["placement"]
    if placement not in settings["supported_placements"]:
        return {"status": "retest", "reasons": ["Sway requires a configured trunk placement, such as lower_back or chest."]}
    if (settings["require_horizontal_alignment_confirmation"]
            and recording.metadata.get("horizontal_alignment_confirmed") is not True):
        return {"status": "retest", "reasons": ["Confirm that configured sway axes are horizontal for this mounting before analysis."]}
    quality = assess_imu(recording, "sway", config)
    if quality.status == "retest":
        return quality.retest_response()
    _validate(config, float(recording.metadata["sampling_rate_hz"]))
    rows, expected = _windows(recording, config)
    count = len(rows)
    coverage = count / expected if expected else 0.0
    if count < config["uncertainty"]["minimum_valid_windows"] or coverage < settings["minimum_valid_window_fraction"]:
        return {"status": "retest", "reasons": [f"Too few complete sway windows after exclusions: {count} valid of {expected} expected."]}
    per_window = [{"start_s": start, "segment_id": segment_id,
                   "features": _metrics(values, float(config["preprocess"]["resample_hz"]))}
                  for start, values, segment_id, _ in rows]
    features, spread = {}, {}
    for name in per_window[0]["features"]:
        values = [row["features"][name] for row in per_window if row["features"][name] is not None]
        features[name] = float(np.median(values)) if values else None
        spread[name] = {"iqr": np.percentile(values, [25, 75]).tolist() if values else None,
                        "n_valid_windows": len(values), "quality_score": quality.score}
    # Overlap may repeat an edge in several windows. Sum each acceleration edge
    # once; keys preserve separate segments, never manufacturing a gap edge.
    edges = {}
    for _, values, segment_id, offset in rows:
        distances = np.linalg.norm(np.diff(values, axis=0), axis=1)
        for sample_offset, distance in enumerate(distances):
            edges[(segment_id, offset + sample_offset)] = float(distance)
    features["acceleration_path_length_proxy_m_s2"] = float(sum(edges.values()))
    return {"status": "ok", "quality": quality.as_dict(), "features": features,
            "per_window": per_window, "spread": spread, "n_valid_windows": count,
            "expected_windows": expected, "placement": placement,
            "observed_duration_s": len(edges) / float(config["preprocess"]["resample_hz"]),
            "horizontal_axes": list(settings["horizontal_axes"]),
            "limitations": [
                "These are acceleration-domain proxies, not body displacement, center-of-pressure motion, or physical sway path.",
                "The ellipse is a nominal 95% Gaussian covariance contour, not a confidence interval or guaranteed 95% empirical containment.",
                "Mounting and horizontal alignment are assumed confirmed; dynamic tilt changes gravity projection and can mimic sway.",
                "Global path totals unique acceleration edges covered by windows; window IQRs describe per-window paths, not uncertainty in that total.",
                "Path length depends on filtering, sampling and observed duration; only compare recordings using the same settings.",
                "Overlapping-window IQRs describe spread, not independent observations or calibrated clinical uncertainty."
            ]}
