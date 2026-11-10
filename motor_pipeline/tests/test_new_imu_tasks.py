"""Cross-module checks using the public synthetic sources and real config."""

import json

import pytest

from motor_pipeline.config import load_config
from motor_pipeline.features.reps import extract_reps
from motor_pipeline.features.sway import extract_sway
from motor_pipeline.io import normalize_imu
from motor_pipeline.synthetic import postural_sway, repetitions


@pytest.mark.parametrize("fs", [50, 100, 200])
def test_sway_fixture_works_at_supported_source_rates(fs):
    cfg = load_config()
    cfg["input"]["sampling_rate_hz"] = fs
    cfg["input"]["allow_missing_gyroscope"] = True
    frame = postural_sway(sampling_rate_hz=fs, noise_std=0).drop(columns=["gx", "gy", "gz"])
    recording = normalize_imu(frame, cfg, {"placement": "lower_back", "horizontal_alignment_confirmed": True})
    result = extract_sway(recording, cfg)
    assert result["status"] == "ok"
    assert result["features"]["mean_frequency_hz"] == pytest.approx(.5, abs=.04)
    assert result["features"]["horizontal_acceleration_rms_m_s2"] == pytest.approx(.08246, rel=.08)
    json.dumps(result, allow_nan=False)


@pytest.mark.parametrize("fs", [50, 100, 200])
def test_repetition_fixture_recovers_range_and_period(fs):
    cfg = load_config()
    cfg["input"]["sampling_rate_hz"] = fs
    recording = normalize_imu(repetitions(sampling_rate_hz=fs), cfg)
    before = json.dumps(cfg, sort_keys=True)
    result = extract_reps(recording, cfg)
    assert result["status"] == "ok"
    assert result["features"]["mean_rep_duration_s"] == pytest.approx(1.5, abs=.03)
    assert result["features"]["range_of_motion_proxy"] == pytest.approx(40, rel=.08)
    assert json.dumps(cfg, sort_keys=True) == before
    json.dumps(result, allow_nan=False)


def test_repetition_slowing_fixture_has_known_period_slope():
    recording = normalize_imu(repetitions(period_start_s=1, period_end_s=2, noise_std=0))
    result = extract_reps(recording)
    assert result["status"] == "ok"
    assert result["features"]["duration_slope_s_per_s"] == pytest.approx(.025, abs=.005)
    assert result["features"]["slowing_fraction"] > .5


def test_unconfirmed_sway_mounting_cannot_silently_use_default_axes():
    recording = normalize_imu(postural_sway(), metadata={"placement": "lower_back"})
    result = extract_sway(recording)
    assert result["status"] == "retest"
    assert "features" not in result
