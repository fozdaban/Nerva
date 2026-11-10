"""Wrist tap proxies must preserve timing/decrement and distinguish data gaps."""

import json

import numpy as np
import pytest

from motor_pipeline.config import load_config
from motor_pipeline.features import extract_tapping_imu
from motor_pipeline.io import normalize_imu
from motor_pipeline.synthetic import add_artifact, resting_noise, tapping


def extract(frame, cfg=None, placement="wrist"):
    cfg = load_config() if cfg is None else cfg
    return extract_tapping_imu(normalize_imu(frame, cfg, {"placement": placement}), cfg)


@pytest.mark.parametrize("fs", [50, 100, 200])
@pytest.mark.parametrize("rate", [1.5, 2, 3, 4])
def test_known_rate_and_regular_intervals(fs, rate):
    cfg = load_config()
    cfg["input"]["sampling_rate_hz"] = fs
    result = extract(tapping(sampling_rate_hz=fs, rate_hz=rate), cfg)
    assert result["status"] == "ok"
    assert result["features"]["tap_rate_hz"] == pytest.approx(rate, rel=.02)
    assert result["features"]["inter_tap_interval_cv"] < .05
    assert result["features"]["hesitation_count"] == 0


def test_deduplicated_count_matches_known_pulse_times():
    result = extract(tapping(rate_hz=2, decrement_fraction=0, noise_std=0))
    # Source taps .25,.75,...,19.75; accepted windows cover .5–18.5 s.
    assert result["features"]["tap_count"] == 36
    assert len(result["event_times_s"]) == 36
    assert sum(row["features"]["tap_count"] for row in result["per_window"]) > 36


@pytest.mark.parametrize("decrement", [0, .3, .6])
def test_amplitude_decrement_over_observed_span(decrement):
    result = extract(tapping(rate_hz=2, decrement_fraction=decrement, noise_std=0))
    times = result["event_times_s"]
    # Forty source pulse amplitudes decline linearly from 1 to 1-decrement.
    source_first, source_last = .25, 19.75
    first_amplitude = 1 - decrement * (times[0] - source_first) / (source_last - source_first)
    last_amplitude = 1 - decrement * (times[-1] - source_first) / (source_last - source_first)
    expected = 1 - last_amplitude / first_amplitude
    assert result["features"]["amplitude_decrement_fraction"] == pytest.approx(expected, abs=.03)
    if decrement:
        assert result["features"]["amplitude_slope_per_s"] < 0


def test_increasing_amplitude_is_not_forced_to_decrement():
    frame = tapping(decrement_fraction=0, noise_std=0)
    for axis in ("ax", "gx"):
        frame[axis] *= 1 + frame.t / 20
    result = extract(frame)
    assert result["features"]["amplitude_slope_per_s"] > 0
    assert result["features"]["amplitude_decrement_fraction"] < 0


def test_long_pause_is_one_hesitation_not_repeated_window_counts():
    frame = tapping(duration_s=30, rate_hz=2, decrement_fraction=0, noise_std=0)
    frame.loc[(frame.t >= 10) & (frame.t < 13), ["ax", "gx"]] = 0
    result = extract(frame)
    assert result["status"] == "ok"
    assert result["features"]["hesitation_count"] == 1
    assert max(result["inter_tap_intervals_s"]) == pytest.approx(3.5, abs=.03)
    assert result["features"]["inter_tap_interval_cv"] > .4


def test_missing_data_is_not_a_hesitation_or_an_amplitude_trend():
    frame = tapping(duration_s=30, rate_hz=2, decrement_fraction=0, noise_std=0)
    frame.loc[1400:1409, "gx"] = np.nan
    result = extract(frame)
    assert result["status"] == "ok"
    assert len(set(result["event_segment_ids"])) == 2
    assert result["features"]["hesitation_count"] == 0
    assert max(result["inter_tap_intervals_s"]) < .53
    assert result["features"]["amplitude_slope_per_s"] is None
    assert result["features"]["amplitude_decrement_fraction"] is None


def test_fixed_rotation_and_polarity_preserve_timing_and_amplitude():
    frame = tapping(noise_std=0)
    rotated = frame.copy()
    for axes in (["ax", "ay", "az"], ["gx", "gy", "gz"]):
        rotated[axes] = -frame[axes].to_numpy()[:, [1, 2, 0]]
    first, second = extract(frame), extract(rotated)
    for key in ("tap_rate_hz", "tap_count", "median_tap_amplitude", "amplitude_decrement_fraction"):
        assert second["features"][key] == pytest.approx(first["features"][key], rel=1e-5)


def test_gyro_selection_and_units():
    cfg = load_config()
    cfg["features"]["tapping_imu"].update(primary_sensor="gyro", axis="gx")
    result = extract(tapping(rate_hz=2, noise_std=0), cfg)
    assert result["status"] == "ok"
    assert result["amplitude_unit"] == "deg/s"
    assert result["features"]["tap_rate_hz"] == pytest.approx(2, rel=.02)


@pytest.mark.parametrize("artifact", ["gap", "missing", "clipping", "duplicate", "out_of_order"])
def test_bad_quality_retests_without_features(artifact):
    result = extract(add_artifact(tapping(), artifact, duration_s=2))
    assert result["status"] == "retest"
    assert set(result) == {"status", "reasons"}


@pytest.mark.parametrize("generator", [resting_noise, lambda: tapping(amplitude=.001)])
def test_noise_and_too_small_signal_retest(generator):
    assert extract(generator())["status"] == "retest"


def test_fragmentation_and_wrong_placement_retest():
    frame = tapping()
    frame.loc[np.arange(200, 2000, 200), "gx"] = np.nan
    assert extract(frame)["status"] == "retest"
    assert extract(tapping(), placement="ankle")["status"] == "retest"


def test_missing_required_channel_retests():
    cfg = load_config()
    cfg["input"]["allow_missing_gyroscope"] = True
    assert extract(tapping().drop(columns=["gx", "gy", "gz"]), cfg)["status"] == "retest"


def test_json_safe_output_spread_and_input_immutability():
    recording = normalize_imu(tapping())
    before = recording.data.copy(deep=True)
    result = extract_tapping_imu(recording)
    assert recording.data.equals(before)
    json.dumps(result, allow_nan=False)
    assert set(result["features"]) == set(result["spread"])
    assert result["spread"]["tap_rate_hz"]["n_valid_windows"] == result["n_valid_windows"]
    assert result["spread"]["tap_rate_hz"]["quality_score"] == result["quality"]["score"]


@pytest.mark.parametrize("key,value", [("primary_sensor", "unknown"), ("axis", "gx"),
                                       ("minimum_peak_distance_s", 0), ("event_merge_tolerance_s", .2),
                                       ("minimum_peaks_per_window", 2), ("minimum_valid_window_fraction", 2),
                                       ("hesitation_median_multiplier", 1)])
def test_invalid_config_raises(key, value):
    cfg = load_config()
    cfg["features"]["tapping_imu"][key] = value
    with pytest.raises(ValueError):
        extract(tapping(), cfg)
