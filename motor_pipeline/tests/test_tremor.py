"""Frequency/amplitude recovery, orientation invariance and honest abstention."""

import json

import numpy as np
import pytest

from motor_pipeline.config import load_config
from motor_pipeline.features import extract_tremor
from motor_pipeline.io import normalize_imu
from motor_pipeline.synthetic import add_artifact, resting_noise, tremor


def extract(frame, cfg=None):
    cfg = load_config() if cfg is None else cfg
    return extract_tremor(normalize_imu(frame, cfg), cfg)


@pytest.mark.parametrize("fs", [50, 100, 200])
@pytest.mark.parametrize("frequency", [3, 3.2, 5, 5.1, 7, 11.8, 12])
def test_recovers_known_frequency_within_point_two_hz(fs, frequency):
    cfg = load_config()
    cfg["input"]["sampling_rate_hz"] = fs
    result = extract(tremor(sampling_rate_hz=fs, frequency_hz=frequency), cfg)
    assert result["status"] == "ok"
    assert result["features"]["dominant_freq_hz"] == pytest.approx(frequency, abs=.2)
    assert result["features"]["accel_dominant_freq_hz"] == pytest.approx(frequency, abs=.2)
    assert result["features"]["tremor_present_fraction"] > .9
    assert result["features"]["gyro_peak_frequency_sd_hz"] < .1


def test_recovers_power_and_vector_rms_in_physical_units():
    result = extract(tremor(noise_std=0, accel_amplitude=.5, gyro_amplitude=10))
    f = result["features"]
    assert f["gyro_power_band"] == pytest.approx(50, rel=.03)
    assert f["gyro_power_narrow"] == pytest.approx(50, rel=.03)
    assert f["accel_power_band"] == pytest.approx(.125, rel=.03)
    assert f["gyro_rms"] == pytest.approx(10 / np.sqrt(2), rel=.03)
    assert f["gyro_band_power_ratio"] > .99


def test_narrow_and_wide_band_power_are_distinct():
    result = extract(tremor(frequency_hz=10, noise_std=0))
    assert result["features"]["gyro_power_narrow"] < 1
    assert result["features"]["gyro_power_band"] == pytest.approx(50, rel=.03)


def test_fixed_rotation_preserves_frequency_power_and_rms():
    frame = tremor(noise_std=0)
    axis = np.array([1, 2, 3]) / np.linalg.norm([1, 2, 3])
    angle = .8
    cross = np.array([[0, -axis[2], axis[1]], [axis[2], 0, -axis[0]], [-axis[1], axis[0], 0]])
    rotation = np.eye(3) * np.cos(angle) + (1 - np.cos(angle)) * np.outer(axis, axis) + np.sin(angle) * cross
    rotated = frame.copy()
    for channels in (["ax", "ay", "az"], ["gx", "gy", "gz"]):
        rotated[channels] = frame[channels].to_numpy() @ rotation.T
    first, second = extract(frame)["features"], extract(rotated)["features"]
    for feature in ("dominant_freq_hz", "gyro_power_band", "accel_power_band", "gyro_rms", "accel_rms"):
        assert second[feature] == pytest.approx(first[feature], rel=1e-5)


def test_magnitude_rectification_does_not_double_primary_frequency():
    result = extract(tremor(frequency_hz=5, noise_std=0))
    assert result["features"]["dominant_freq_hz"] == pytest.approx(5, abs=.2)
    assert result["features"]["gyro_magnitude_power_band"] > 1  # supplemental 10 Hz evidence


@pytest.mark.parametrize("generator", [resting_noise, lambda: tremor(accel_amplitude=.001, gyro_amplitude=.01)])
def test_noise_or_low_amplitude_does_not_get_a_frequency(generator):
    result = extract(generator())
    assert result["status"] == "ok"
    assert result["features"]["dominant_freq_hz"] is None
    assert result["features"]["tremor_present_fraction"] == 0
    assert result["spread"]["dominant_freq_hz"]["n_valid_windows"] == 0


def test_static_sensor_has_zero_power_not_random_frequency():
    result = extract(resting_noise(accel_noise_std=0, gyro_noise_std=0))
    assert result["status"] == "ok"
    assert result["features"]["dominant_freq_hz"] is None
    assert result["features"]["gyro_power_band"] == 0
    json.dumps(result, allow_nan=False)


def test_broadband_denominator_not_prefiltered_to_tremor_band():
    frame = tremor(noise_std=0)
    frame["gx"] += 50 * np.sin(2 * np.pi * 20 * frame.t)
    result = extract(frame)
    assert result["features"]["gyro_band_power_ratio"] < .1
    assert result["features"]["dominant_freq_hz"] is None


def test_episodic_fraction_is_mean_not_median():
    frame = tremor(duration_s=30, noise_std=0)
    frame.loc[frame.t >= 10, ["ax", "gx"]] = 0
    result = extract(frame)
    flags = [window["features"]["gyro_tremor_present"] for window in result["per_window"]]
    assert 0 < result["features"]["tremor_present_fraction"] < 1
    assert result["features"]["tremor_present_fraction"] == np.mean(flags)


@pytest.mark.parametrize("artifact", ["gap", "missing", "clipping", "duplicate", "out_of_order"])
def test_quality_rejection_has_no_features(artifact):
    result = extract(add_artifact(tremor(), artifact, duration_s=2))
    assert result["status"] == "retest"
    assert set(result) == {"status", "reasons"}


def test_missing_gyro_retests():
    cfg = load_config()
    cfg["input"]["allow_missing_gyroscope"] = True
    result = extract(tremor().drop(columns=["gx", "gy", "gz"]), cfg)
    assert result["status"] == "retest"


def test_quality_pass_with_fragmented_samples_still_retests():
    frame = tremor()
    frame.loc[np.arange(200, 2000, 200), "gx"] = np.nan  # <1%, but no complete windows
    result = extract(frame)
    assert result["status"] == "retest"
    assert "complete tremor windows" in result["reasons"][0]


def test_small_gap_exclusion_preserves_enough_other_windows():
    frame = tremor(duration_s=30)
    frame.loc[1400:1409, "gx"] = np.nan
    result = extract(frame)
    assert result["status"] == "ok"
    assert result["n_valid_windows"] < result["expected_windows"]
    for window in result["per_window"]:
        start = window["start_s"]
        assert start + 4 <= 14 or start >= 14.1


def test_window_spread_is_not_mislabeled_as_ci():
    result = extract(tremor())
    spread = result["spread"]["gyro_rms"]
    assert set(spread) == {"iqr", "n_valid_windows", "quality_score"}
    assert spread["iqr"][0] <= spread["iqr"][1]
    assert spread["n_valid_windows"] == result["n_valid_windows"]
    assert spread["quality_score"] == result["quality"]["score"]
    assert not any("ci95" in name for name in spread)
    json.dumps(result, allow_nan=False)


@pytest.mark.parametrize("key,value", [("primary_sensor", "unknown"), ("minimum_band_ratio", 2),
                                       ("minimum_valid_window_fraction", 0), ("welch_segment_s", .2),
                                       ("narrow_band_hz", [1, 7])])
def test_bad_feature_config_fails_explicitly(key, value):
    cfg = load_config()
    cfg["features"]["tremor"][key] = value
    with pytest.raises(ValueError):
        extract(tremor(), cfg)
