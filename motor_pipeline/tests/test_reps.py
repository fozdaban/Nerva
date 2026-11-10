"""Complete-cycle repetition metrics and trend recovery from known signals."""

import json

import numpy as np
import pytest

from motor_pipeline.config import load_config
from motor_pipeline.features.reps import extract_reps
from motor_pipeline.io import normalize_imu
from motor_pipeline.synthetic import add_artifact, repetitions, resting_noise, walking


def extract(frame, cfg=None, placement="wrist"):
    cfg = load_config() if cfg is None else cfg
    return extract_reps(normalize_imu(frame, cfg, {"placement": placement}), cfg)


@pytest.mark.parametrize("fs", [50, 100, 200])
@pytest.mark.parametrize("period", [.75, 1, 1.5, 2])
def test_period_duration_and_consistency(fs, period):
    cfg = load_config()
    cfg["input"]["sampling_rate_hz"] = fs
    result = extract(walking(sampling_rate_hz=fs, cadence_steps_min=120 / period), cfg)
    assert result["status"] == "ok"
    assert result["features"]["mean_rep_duration_s"] == pytest.approx(period, abs=.025)
    assert result["features"]["rep_duration_cv"] < .03
    assert abs(result["features"]["duration_slope_s_per_s"]) < .003


def test_complete_count_excludes_partial_reps_and_window_duplicates():
    result = extract(walking(noise_std=0))
    # .5–28.5 s prepared coverage contains 28 positive peaks: 27 full cycles.
    assert result["features"]["rep_count"] == 27
    assert len(result["events"]) == 28
    assert sum(row["features"]["rep_count"] for row in result["per_window"]) > 27


def test_gyro_range_proxy_matches_integrated_sinusoid():
    result = extract(walking(noise_std=0, gyro_amplitude=40))
    assert result["range_of_motion_unit"] == "deg"
    assert result["features"]["range_of_motion_proxy"] == pytest.approx(40 / np.pi, rel=.04)
    assert result["features"]["range_of_motion_cv"] < .03


def test_acceleration_excursion_is_not_labeled_joint_angle():
    cfg = load_config()
    cfg["features"]["reps"].update(primary_sensor="accel", axis="ax")
    result = extract(walking(noise_std=0), cfg)
    assert result["range_of_motion_unit"] == "m/s2"
    assert result["features"]["range_of_motion_proxy"] == pytest.approx(1, rel=.04)
    assert result["features"]["mean_rep_duration_s"] == pytest.approx(1, abs=.02)


def test_slowing_duration_slope_preserved():
    frame = walking(duration_s=40, noise_std=0)
    slope, initial = .02, .8
    phase = 2 * np.pi * np.log(1 + slope * frame.t / initial) / slope
    frame["gx"] = 40 * np.sin(phase)
    frame["ax"] = .5 * np.sin(phase)
    frame["az"] = 9.80665 + np.cos(2 * phase)
    result = extract(frame)
    assert result["status"] == "ok"
    assert result["features"]["duration_slope_s_per_s"] == pytest.approx(slope, abs=.003)
    assert result["features"]["slowing_fraction"] > .5


def test_speeding_up_remains_negative_slowing():
    frame = walking(duration_s=30, noise_std=0)
    slope, initial = -.015, 1.3
    phase = 2 * np.pi * np.log(1 + slope * frame.t / initial) / slope
    frame["gx"] = 40 * np.sin(phase)
    frame["ax"] = .5 * np.sin(phase)
    frame["az"] = 9.80665 + np.cos(2 * phase)
    result = extract(frame)
    assert result["status"] == "ok"
    assert result["features"]["duration_slope_s_per_s"] == pytest.approx(slope, abs=.003)
    assert result["features"]["slowing_fraction"] < 0


def test_variable_repetitions_keep_cv():
    frame = walking(duration_s=40, noise_std=0)
    boundaries = np.r_[0, np.cumsum(np.tile([.85, 1.15], 25))]
    phase = np.interp(frame.t, boundaries, np.arange(len(boundaries))) * 2 * np.pi
    frame["gx"] = 40 * np.cos(phase)
    frame["ax"] = .5 * np.cos(phase)
    result = extract(frame)
    assert result["status"] == "ok"
    assert result["features"]["rep_duration_cv"] == pytest.approx(.15, abs=.035)


@pytest.mark.parametrize("period,duration", [(4, 20), (6, 30)])
def test_short_slow_constant_task_withholds_boundary_biased_trend(period, duration):
    result = extract(repetitions(duration_s=duration, period_start_s=period, noise_std=0))
    assert result["status"] == "ok"
    assert result["features"]["mean_rep_duration_s"] == pytest.approx(period, abs=.1)
    assert result["features"]["duration_slope_s_per_s"] is None
    assert result["features"]["slowing_fraction"] is None


@pytest.mark.parametrize("period,duration", [(4, 40), (6, 60), (4, 80), (6, 120)])
def test_longer_slow_constant_task_has_no_large_spurious_slowing(period, duration):
    result = extract(repetitions(duration_s=duration, period_start_s=period, noise_std=0))
    assert result["status"] == "ok"
    assert result["features"]["duration_slope_s_per_s"] is not None
    assert abs(result["features"]["duration_slope_s_per_s"]) < .003
    assert abs(result["features"]["slowing_fraction"]) < .02


def test_cycle_support_excludes_late_subthreshold_motion():
    frame = walking(duration_s=60, noise_std=0)
    time = frame.t.to_numpy()
    gyro = np.where(time < 32, 8 * np.sin(2 * np.pi * time), 0)
    for center in np.arange(33, 60, 2):
        gyro += 7 * np.exp(-.5 * ((time - center) / .06) ** 2)
    frame["gx"] = gyro
    result = extract(frame)
    assert result["status"] == "ok"
    accepted_end = max(window["end_s"] for window in result["per_window"])
    assert accepted_end < 40
    assert max(cycle["end_s"] for cycle in result["cycles"]) <= accepted_end
    assert abs(result["features"]["slowing_fraction"]) < .02
    # Every part of each cycle lies inside the accepted window UNION, not
    # merely somewhere in a source segment that also had valid motion.
    for cycle in result["cycles"]:
        for instant in np.linspace(cycle["start_s"], cycle["end_s"], 101):
            assert any(window["start_s"] - 1e-9 <= instant <= window["end_s"] + 1e-9
                       for window in result["per_window"])


def test_uncovered_hole_splits_runs_even_within_one_source_segment():
    frame = walking(duration_s=60, noise_std=0)
    frame.loc[(frame.t >= 18) & (frame.t < 28), "gx"] = 0
    result = extract(frame)
    assert result["status"] == "ok"
    assert len({run["segment_id"] for run in result["analysis_runs"]}) == 1
    assert len(result["analysis_runs"]) == 2
    assert result["features"]["duration_slope_s_per_s"] is None
    assert result["features"]["slowing_fraction"] is None
    for cycle in result["cycles"]:
        assert cycle["end_s"] < 22 or cycle["start_s"] > 24


def test_fixed_rotation_keeps_period_and_range():
    frame = walking(noise_std=0)
    rotated = frame.copy()
    for axes in (["ax", "ay", "az"], ["gx", "gy", "gz"]):
        rotated[axes] = frame[axes].to_numpy()[:, [1, 2, 0]]
    first, second = extract(frame), extract(rotated)
    for key in ("mean_rep_duration_s", "range_of_motion_proxy", "rep_count"):
        assert second["features"][key] == pytest.approx(first["features"][key], rel=1e-5)


def test_signed_axis_is_supported():
    cfg = load_config()
    cfg["features"]["reps"]["axis"] = "-gx"
    result = extract(walking(), cfg)
    assert result["features"]["mean_rep_duration_s"] == pytest.approx(1, abs=.02)


def test_missing_data_never_becomes_a_slow_rep():
    frame = walking()
    frame.loc[1400:1409, "gx"] = np.nan
    result = extract(frame)
    assert result["status"] == "ok"
    assert result["features"]["duration_slope_s_per_s"] is None
    assert result["features"]["slowing_fraction"] is None
    assert max(cycle["duration_s"] for cycle in result["cycles"]) < 1.05
    assert result["features"]["rep_count"] < 27
    for cycle in result["cycles"]:
        assert cycle["end_s"] < 14 or cycle["start_s"] > 14.1


@pytest.mark.parametrize("artifact", ["gap", "clipping", "missing", "duplicate", "out_of_order"])
def test_excess_artifacts_retest_without_features(artifact):
    result = extract(add_artifact(walking(), artifact, duration_s=2))
    assert set(result) == {"status", "reasons"}
    assert result["status"] == "retest"


@pytest.mark.parametrize("frame", [resting_noise(), walking(gyro_amplitude=0), walking(duration_s=5)])
def test_no_signal_or_short_recording_retests(frame):
    assert extract(frame)["status"] == "retest"


def test_missing_gyro_and_unsupported_placement_retest():
    cfg = load_config()
    cfg["input"]["allow_missing_gyroscope"] = True
    assert extract(walking().drop(columns=["gx", "gy", "gz"]), cfg)["status"] == "retest"
    assert extract(walking(), placement="head")["status"] == "retest"


def test_placement_can_be_configured_for_guided_exercise():
    cfg = load_config()
    cfg["features"]["reps"]["accepted_placements"] = ["chest"]
    assert extract(walking(), cfg, placement="chest")["status"] == "ok"


def test_json_spread_and_immutability():
    cfg = load_config()
    before_cutoff = cfg["preprocess"]["gravity_highpass_hz"]
    recording = normalize_imu(walking())
    before = recording.data.copy(deep=True)
    result = extract_reps(recording, cfg)
    assert recording.data.equals(before)
    assert cfg["preprocess"]["gravity_highpass_hz"] == before_cutoff
    json.dumps(result, allow_nan=False)
    assert set(result["spread"]) == set(result["features"])
    assert result["spread"]["mean_rep_duration_s"]["quality_score"] == result["quality"]["score"]


@pytest.mark.parametrize("key,value", [("axis", "ax"), ("primary_sensor", "unknown"),
                                       ("gravity_highpass_hz", .3), ("minimum_peak_distance_s", 0),
                                       ("rep_duration_bounds_s", [1, .5]), ("minimum_valid_window_fraction", 0),
                                       ("accepted_placements", []), ("minimum_trend_cycles", 2),
                                       ("trend_boundary_cycles", 0)])
def test_invalid_config_raises(key, value):
    cfg = load_config()
    cfg["features"]["reps"][key] = value
    with pytest.raises(ValueError):
        extract(walking(), cfg)
