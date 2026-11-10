"""Known math for additional non-clinical sway and repetition fixtures."""

import numpy as np
import pytest
from scipy.integrate import cumulative_trapezoid

from motor_pipeline.synthetic import postural_sway, repetitions


@pytest.mark.parametrize("fs", [50, 100, 200])
def test_sway_ellipse_and_frequency(fs):
    frame = postural_sway(sampling_rate_hz=fs, noise_std=0)
    assert np.sqrt(np.mean(frame.ax ** 2 + frame.ay ** 2)) == pytest.approx(np.sqrt((.1 ** 2 + .06 ** 2) / 2))
    frequencies = np.fft.rfftfreq(len(frame), d=1 / fs)
    peak = frequencies[np.argmax(np.abs(np.fft.rfft(frame.ax)))]
    assert peak == pytest.approx(.5)
    assert frame.az.mean() == pytest.approx(9.80665)


@pytest.mark.parametrize("fs", [50, 100, 200])
def test_repetition_angular_range(fs):
    frame = repetitions(duration_s=30, sampling_rate_hz=fs, period_start_s=1.5, noise_std=0)
    angle = cumulative_trapezoid(frame.gx, dx=1 / fs, initial=0)
    assert np.ptp(angle) == pytest.approx(40, rel=.002)


def test_slowing_fixture_matches_known_period_law():
    frame = repetitions(period_start_s=1, period_end_s=2, noise_std=0)
    t = frame.t.to_numpy()
    phase = 2 * np.pi * np.log(1 + t / 40) * 40
    np.testing.assert_allclose(frame.ax, .5 * np.sin(phase), atol=1e-12)


@pytest.mark.parametrize("generator,parameters", [
    (postural_sway, {"amplitude_x": -1}), (postural_sway, {"frequency_hz": 0}),
    (repetitions, {"period_start_s": 0}), (repetitions, {"period_end_s": -1}),
    (repetitions, {"angular_range_deg": float("nan")}),
])
def test_invalid_parameters(generator, parameters):
    with pytest.raises(ValueError):
        generator(**parameters)
