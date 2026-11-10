"""Filter uninterrupted complete segments; never bridge a long missing interval."""

from dataclasses import dataclass
from fractions import Fraction
from typing import Any, Mapping

import numpy as np
from scipy.signal import butter, resample_poly, sosfiltfilt

from motor_pipeline.io.adapter import AXES, Recording


@dataclass
class PreparedIMU:
    """Window arrays have shape (windows, samples, six canonical axes)."""

    dynamic: np.ndarray
    bandpassed: np.ndarray
    starts_s: list[float]
    sampling_rate_hz: float
    expected_windows: int


def validate_preprocess(config: Mapping[str, Any], source_fs: float, band: list[float]) -> None:
    """Configuration errors are programmer errors, not a poor-data result."""
    p = config["preprocess"]
    fs = p["resample_hz"]
    if not np.isfinite(source_fs) or source_fs <= 0 or not np.isfinite(fs) or fs <= 0:
        raise ValueError("Source and target sample rates must be positive and finite")
    if not isinstance(p["filter_order"], int) or not 1 <= p["filter_order"] <= 8:
        raise ValueError("filter_order must be an integer from 1 to 8")
    if len(band) != 2 or not 0 < p["gravity_highpass_hz"] < band[0] < band[1] < min(source_fs, fs) / 2:
        raise ValueError("Filter bands must lie above gravity cutoff and below source/target Nyquist")
    if not np.isfinite(p["window_s"]) or not 2 <= p["window_s"] <= 5:
        raise ValueError("window_s must be between 2 and 5 seconds")
    if not np.isfinite(p["overlap_fraction"]) or not 0 <= p["overlap_fraction"] < 1:
        raise ValueError("overlap_fraction must be in [0, 1)")
    for key in ("edge_trim_s", "maximum_interpolation_gap_s"):
        if not np.isfinite(p[key]) or p[key] <= 0:
            raise ValueError(f"{key} must be finite and positive")
    if round(p["window_s"] * fs) < 4:
        raise ValueError("Target sample rate provides too few samples per window")


def prepare_imu(recording: Recording, config: Mapping[str, Any], band: list[float]) -> PreparedIMU:
    """Resample with anti-aliasing, remove constant gravity, filter and window.

    Nonfinite/clipped rows and time discontinuities break segments. Segments are
    filtered independently; padding transients are trimmed at both ends. Input
    must have passed task quality checks. No input values are modified.
    """
    p = config["preprocess"]
    native_fs = float(recording.metadata["sampling_rate_hz"])
    validate_preprocess(config, native_fs, band)
    fs = float(p["resample_hz"])
    t = recording.data.t.to_numpy(dtype=float)
    values = recording.data[list(AXES)].to_numpy(dtype=float)
    if len(t) < 2 or not np.isfinite(t).all() or np.any(np.diff(t) <= 0):
        raise ValueError("Preprocessing requires finite, increasing canonical timestamps")
    hardware = config["input"]
    limits = np.array([hardware["acceleration_range_g"] * hardware["standard_gravity_m_s2"]] * 3
                      + [hardware["gyroscope_range_deg_s"]] * 3)
    usable = (np.isfinite(values).all(axis=1)
              & (np.abs(values) < limits * config["quality"]["clipping_range_fraction"]).all(axis=1))
    indices = np.flatnonzero(usable)
    # Always permit nominal spacing, even at 50 Hz with a smaller gap setting.
    max_gap = max(p["maximum_interpolation_gap_s"], 1.5 / native_fs)
    cuts = np.flatnonzero((np.diff(indices) > 1) | (np.diff(t[indices]) > max_gap + 1e-9)) + 1
    segments = np.split(indices, cuts)
    width = int(round(p["window_s"] * fs))
    hop = max(1, int(round(width * (1 - p["overlap_fraction"]))))
    trim = int(np.ceil(p["edge_trim_s"] * fs))
    total_samples = int(round((t[-1] - t[0] + 1 / native_fs) * fs))
    expected = max(0, 1 + (total_samples - 2 * trim - width) // hop)
    # Second-order sections implement forward/backward zero-phase filtering
    # without ill-conditioned high-order polynomial coefficients.
    highpass = butter(p["filter_order"], p["gravity_highpass_hz"], btype="highpass", fs=fs, output="sos")
    bandpass = butter(p["filter_order"], band, btype="bandpass", fs=fs, output="sos")
    ratio = Fraction(str(fs)) / Fraction(str(native_fs))
    ratio = ratio.limit_denominator(1000)
    dynamic_windows, band_windows, starts = [], [], []
    for segment in segments:
        if len(segment) < 2:
            continue
        start, end = t[segment[0]], t[segment[-1]]
        if end - start + 1 / native_fs < p["window_s"] + 2 * p["edge_trim_s"] - 1e-9:
            continue
        native_t = start + np.arange(int(np.floor((end - start) * native_fs + 1e-7)) + 1) / native_fs
        regular = np.column_stack([np.interp(native_t, t[segment], values[segment, axis]) for axis in range(6)])
        # Removing DC before resampling avoids a gravity step at padded edges.
        regular -= regular.mean(axis=0)
        resampled = resample_poly(regular, ratio.numerator, ratio.denominator, axis=0)
        # resample_poly can emit a padded sample beyond the final native sample.
        count = int(np.floor((end - start) * fs + 1e-7)) + 1
        resampled = resampled[:count]
        if len(resampled) < width + 2 * trim:
            continue
        dynamic = sosfiltfilt(highpass, resampled, axis=0)
        filtered = sosfiltfilt(bandpass, dynamic, axis=0)
        for offset in range(trim, len(resampled) - trim - width + 1, hop):
            dynamic_windows.append(dynamic[offset:offset + width])
            band_windows.append(filtered[offset:offset + width])
            starts.append(float(start + offset / fs))
    shape = (0, width, 6)
    return PreparedIMU(np.stack(dynamic_windows) if dynamic_windows else np.empty(shape),
                       np.stack(band_windows) if band_windows else np.empty(shape),
                       starts, fs, expected)
