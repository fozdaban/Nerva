import numpy as np
import pytest

from motor_pipeline.io import normalize_imu
from motor_pipeline.synthetic import add_artifact, resting_noise, tapping, tremor, walking


def dominant_frequency(values, fs):
    centered = np.asarray(values) - np.mean(values)
    power = np.abs(np.fft.rfft(centered)) ** 2
    frequencies = np.fft.rfftfreq(len(centered), d=1 / fs)
    return frequencies[np.argmax(power[1:]) + 1]


def positive_peaks(values, threshold):
    values = np.asarray(values)
    return np.flatnonzero((values[1:-1] > values[:-2]) & (values[1:-1] >= values[2:])
                         & (values[1:-1] > threshold)) + 1


@pytest.mark.parametrize("fs", [50, 100, 200])
@pytest.mark.parametrize("frequency", [3.2, 5, 11.4])
def test_known_tremor_frequency_and_amplitude(fs, frequency):
    frame = tremor(sampling_rate_hz=fs, frequency_hz=frequency,
                   accel_amplitude=.5, gyro_amplitude=10, noise_std=0)
    assert dominant_frequency(frame.ax, fs) == pytest.approx(frequency, abs=.05)
    assert dominant_frequency(frame.gx, fs) == pytest.approx(frequency, abs=.05)
    assert np.sqrt(np.mean(frame.gx ** 2)) == pytest.approx(10 / np.sqrt(2), abs=.01)


def test_seeded_noise_and_gravity():
    first = resting_noise(seed=9)
    assert first.equals(resting_noise(seed=9))
    assert not first.equals(resting_noise(seed=10))
    assert first.az.mean() == pytest.approx(9.80665, abs=.001)
    assert first.ax.std() == pytest.approx(.01, rel=.1)


@pytest.mark.parametrize("fs", [50, 100, 200])
def test_tapping_rate_and_decrement(fs):
    frame = tapping(sampling_rate_hz=fs, rate_hz=2, decrement_fraction=.4, noise_std=0)
    peaks = positive_peaks(frame.ax, .2)
    assert len(peaks) == 40
    assert 1 / np.median(np.diff(frame.t.iloc[peaks])) == pytest.approx(2, abs=.02)
    assert frame.ax.iloc[peaks[-1]] / frame.ax.iloc[peaks[0]] == pytest.approx(.6, abs=.01)


@pytest.mark.parametrize("cadence", [80, 120, 160])
def test_walking_single_limb_stride_convention(cadence):
    frame = walking(cadence_steps_min=cadence, noise_std=0)
    peaks = positive_peaks(frame.gx, 20)
    recovered_cadence = 120 / np.median(np.diff(frame.t.iloc[peaks]))
    assert recovered_cadence == pytest.approx(cadence, abs=2)
    assert dominant_frequency(frame.gx, 100) == pytest.approx(cadence / 120, abs=.04)


@pytest.mark.parametrize("kind", ["gap", "clipping", "sudden_movement", "missing", "duplicate", "out_of_order"])
def test_artifact_effects_and_no_mutation(kind):
    original = resting_noise()
    snapshot = original.copy(deep=True)
    frame = add_artifact(original, kind)
    assert original.equals(snapshot)
    result = normalize_imu(frame)
    if kind == "gap":
        assert len(result.diagnostics["gaps"]) == 1
        assert result.diagnostics["estimated_missing_samples"] == pytest.approx(100)
    elif kind == "clipping":
        assert frame.ax.max() == pytest.approx(8 * 9.80665)
    elif kind == "sudden_movement":
        assert frame.ax.max() > 49
    elif kind == "missing":
        assert result.diagnostics["nonfinite_sensor_values"] == 100
    elif kind == "duplicate":
        assert result.diagnostics["duplicate_timestamps"] == 100
    else:
        assert result.diagnostics["out_of_order_transitions"] == 99


@pytest.mark.parametrize("factory,kwargs", [
    (resting_noise, {"duration_s": 0}), (resting_noise, {"sampling_rate_hz": 201}),
    (resting_noise, {"accel_noise_std": -1}), (tremor, {"frequency_hz": 50}),
    (tapping, {"rate_hz": 0}), (tapping, {"decrement_fraction": 2}),
    (walking, {"cadence_steps_min": -1}),
])
def test_invalid_generator_parameters(factory, kwargs):
    with pytest.raises(ValueError):
        factory(**kwargs)


def test_invalid_artifact_interval():
    with pytest.raises(ValueError, match="no samples"):
        add_artifact(resting_noise(), "gap", start_s=100)
