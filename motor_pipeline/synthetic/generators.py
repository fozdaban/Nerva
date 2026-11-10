"""Known-frequency fixtures in canonical units for algorithm validation.

Signals include gravity on az. Tremor uses a signed single-axis oscillation:
taking its magnitude before frequency analysis can double the frequency.
"""

from typing import Literal

import numpy as np
import pandas as pd

from motor_pipeline.config import load_config


def _time(duration_s: float, fs: float) -> np.ndarray:
    if not np.isfinite(duration_s) or duration_s <= 0:
        raise ValueError("duration_s must be finite and positive")
    if not np.isfinite(fs) or not 50 <= fs <= 200:
        raise ValueError("sampling_rate_hz must be between 50 and 200")
    count = int(round(duration_s * fs))
    if count < 2:
        raise ValueError("At least two samples are required")
    return np.arange(count, dtype=float) / fs


def _nonnegative(**parameters: float) -> None:
    for name, value in parameters.items():
        if not np.isfinite(value) or value < 0:
            raise ValueError(f"{name} must be finite and nonnegative")


def resting_noise(
    duration_s: float = 20,
    sampling_rate_hz: float = 100,
    accel_noise_std: float = 0.01,
    gyro_noise_std: float = 0.05,
    seed: int = 42,
) -> pd.DataFrame:
    """Independent zero-mean Gaussian sensor noise, with static gravity on z."""
    _nonnegative(accel_noise_std=accel_noise_std, gyro_noise_std=gyro_noise_std)
    t = _time(duration_s, sampling_rate_hz)
    rng = np.random.default_rng(seed)
    acceleration = rng.normal(0, accel_noise_std, (len(t), 3))
    acceleration[:, 2] += load_config()["input"]["standard_gravity_m_s2"]
    gyro = rng.normal(0, gyro_noise_std, (len(t), 3))
    return pd.DataFrame(np.column_stack([t, acceleration, gyro]),
                        columns=["t", "ax", "ay", "az", "gx", "gy", "gz"])


def tremor(
    duration_s: float = 20,
    sampling_rate_hz: float = 100,
    frequency_hz: float = 5,
    accel_amplitude: float = 0.5,
    gyro_amplitude: float = 10,
    noise_std: float = 0.01,
    seed: int = 42,
) -> pd.DataFrame:
    """Sinusoidal ax and gx, with peak amplitudes in m/s² and degrees/s."""
    _nonnegative(accel_amplitude=accel_amplitude, gyro_amplitude=gyro_amplitude)
    if not np.isfinite(frequency_hz) or not 0 < frequency_hz < sampling_rate_hz / 2:
        raise ValueError("frequency_hz must be positive and below Nyquist")
    frame = resting_noise(duration_s, sampling_rate_hz, noise_std, noise_std, seed)
    wave = np.sin(2 * np.pi * frequency_hz * frame.t)
    frame["ax"] += accel_amplitude * wave
    frame["gx"] += gyro_amplitude * wave
    return frame


def tapping(
    duration_s: float = 20,
    sampling_rate_hz: float = 100,
    rate_hz: float = 3,
    amplitude: float = 1,
    decrement_fraction: float = 0.3,
    noise_std: float = 0.005,
    seed: int = 42,
) -> pd.DataFrame:
    """Positive Gaussian ax tap pulses; decrement is first-to-last pulse loss.

    Pulses begin after half a period; wrist angular velocity is a scaled proxy.
    """
    _nonnegative(amplitude=amplitude)
    if not np.isfinite(rate_hz) or not 0 < rate_hz <= sampling_rate_hz / 10:
        raise ValueError("rate_hz must be positive with at least ten samples per tap")
    if not np.isfinite(decrement_fraction) or not 0 <= decrement_fraction <= 1:
        raise ValueError("decrement_fraction must be between zero and one")
    frame = resting_noise(duration_s, sampling_rate_hz, noise_std, noise_std, seed)
    centers = np.arange(0.5 / rate_hz, duration_s - 0.25 / rate_hz, 1 / rate_hz)
    envelope = np.linspace(1, 1 - decrement_fraction, len(centers))
    pulse = np.zeros(len(frame))
    # Evaluate only a local neighbourhood, so generation stays linear in length.
    t = frame.t.to_numpy()
    width = 0.06 / rate_hz
    for center, scale in zip(centers, envelope):
        start = max(0, int((center - 5 * width) * sampling_rate_hz))
        stop = min(len(t), int((center + 5 * width) * sampling_rate_hz) + 1)
        pulse[start:stop] += amplitude * scale * np.exp(-0.5 * ((t[start:stop] - center) / width) ** 2)
    frame["ax"] += pulse
    frame["gx"] += 10 * pulse
    return frame


def walking(
    duration_s: float = 30,
    sampling_rate_hz: float = 100,
    cadence_steps_min: float = 120,
    gyro_amplitude: float = 40,
    noise_std: float = 0.01,
    seed: int = 42,
) -> pd.DataFrame:
    """One ankle: one positive gx mid-swing peak per stride (two steps).

    This convention prevents confusing a one-limb stride rate with step rate.
    It does not model turns, asymmetric gait, or actual walking speed.
    """
    _nonnegative(gyro_amplitude=gyro_amplitude)
    stride_hz = cadence_steps_min / 120
    if not np.isfinite(stride_hz) or not 0 < stride_hz < sampling_rate_hz / 2:
        raise ValueError("cadence_steps_min must give a positive frequency below Nyquist")
    frame = resting_noise(duration_s, sampling_rate_hz, noise_std, noise_std, seed)
    phase = 2 * np.pi * stride_hz * frame.t
    frame["gx"] += gyro_amplitude * np.sin(phase)
    frame["ax"] += 0.5 * np.sin(phase)
    frame["az"] += np.cos(2 * phase)
    return frame


def add_artifact(
    frame: pd.DataFrame,
    kind: Literal["gap", "clipping", "sudden_movement", "missing", "duplicate", "out_of_order"],
    start_s: float = 5,
    duration_s: float = 1,
    axis: str = "ax",
    value: float | None = None,
) -> pd.DataFrame:
    """Copy a fixture and inject an artifact without renumbering timestamps.

    ``value`` is a clipping bound or sudden-movement offset in canonical units.
    Default clipping injects values at the configured sensor full scale.
    """
    if axis not in ("ax", "ay", "az", "gx", "gy", "gz"):
        raise ValueError("axis must name a sensor channel")
    _nonnegative(start_s=start_s)
    if not np.isfinite(duration_s) or duration_s <= 0:
        raise ValueError("duration_s must be finite and positive")
    if value is not None and (not np.isfinite(value) or value <= 0):
        raise ValueError("value must be finite and positive")
    out = frame.copy(deep=True)
    indices = np.flatnonzero(((out.t >= start_s) & (out.t < start_s + duration_s)).to_numpy())
    if not len(indices):
        raise ValueError("Artifact interval contains no samples")
    column = out.columns.get_loc(axis)
    if kind == "gap":
        out = out.iloc[np.setdiff1d(np.arange(len(out)), indices)]
    elif kind == "clipping":
        settings = load_config()["input"]
        limit = value if value is not None else (settings["acceleration_range_g"] * settings["standard_gravity_m_s2"]
                                                  if axis.startswith("a") else settings["gyroscope_range_deg_s"])
        out.iloc[indices, column] = limit
    elif kind == "sudden_movement":
        out.iloc[indices, column] += 50 if value is None else value
    elif kind == "missing":
        out.iloc[indices, column] = np.nan
    elif kind == "duplicate":
        split = indices[-1] + 1
        out = pd.concat([out.iloc[:split], out.iloc[indices], out.iloc[split:]])
    elif kind == "out_of_order":
        if len(indices) < 2:
            raise ValueError("out_of_order needs at least two samples")
        out.iloc[indices] = out.iloc[indices[::-1]].to_numpy()
    else:
        raise ValueError(f"Unknown artifact: {kind}")
    return out.reset_index(drop=True)


def postural_sway(
    duration_s: float = 40,
    sampling_rate_hz: float = 100,
    frequency_hz: float = 0.5,
    amplitude_x: float = 0.1,
    amplitude_y: float = 0.06,
    noise_std: float = 0.001,
    seed: int = 42,
) -> pd.DataFrame:
    """Horizontal acceleration ellipse, not simulated center-of-pressure motion.

    x/y acceleration semi-amplitudes use m/s² and are in quadrature. Placement
    and horizontal-alignment confirmation must be supplied separately to loaders.
    """
    _nonnegative(amplitude_x=amplitude_x, amplitude_y=amplitude_y)
    if not np.isfinite(frequency_hz) or not 0 < frequency_hz < sampling_rate_hz / 2:
        raise ValueError("frequency_hz must be positive and below Nyquist")
    frame = resting_noise(duration_s, sampling_rate_hz, noise_std, noise_std, seed)
    phase = 2 * np.pi * frequency_hz * frame.t
    frame["ax"] += amplitude_x * np.cos(phase)
    frame["ay"] += amplitude_y * np.sin(phase)
    return frame


def repetitions(
    duration_s: float = 40,
    sampling_rate_hz: float = 100,
    period_start_s: float = 1.5,
    period_end_s: float | None = None,
    accel_amplitude: float = 0.5,
    angular_range_deg: float = 40,
    noise_std: float = 0.001,
    seed: int = 42,
) -> pd.DataFrame:
    """One motion cycle per repetition, with an optionally linearly rising period.

    ax is a sinusoidal motion proxy. gx is the derivative of a sine angle with
    peak-to-peak range ``angular_range_deg``, in degrees/s. No anatomical model
    or relationship between that acceleration and displacement is claimed.
    """
    _nonnegative(accel_amplitude=accel_amplitude, angular_range_deg=angular_range_deg)
    period_end_s = period_start_s if period_end_s is None else period_end_s
    if any(not np.isfinite(value) or value <= 0 for value in (period_start_s, period_end_s)):
        raise ValueError("Repetition periods must be finite and positive")
    if 1 / min(period_start_s, period_end_s) >= sampling_rate_hz / 2:
        raise ValueError("Repetition frequency must be below Nyquist")
    frame = resting_noise(duration_s, sampling_rate_hz, noise_std, noise_std, seed)
    t = frame.t.to_numpy()
    slope = (period_end_s - period_start_s) / duration_s
    period = period_start_s + slope * t
    cycles = t / period_start_s if abs(slope) < 1e-12 else np.log(period / period_start_s) / slope
    phase = 2 * np.pi * cycles
    frame["ax"] += accel_amplitude * np.sin(phase)
    frame["gx"] += angular_range_deg * np.pi / period * np.cos(phase)
    return frame
