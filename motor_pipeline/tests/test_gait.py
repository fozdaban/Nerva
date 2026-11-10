"""Known-cycle timing, unique counts, wrist semantics and missing-signal safety."""

import json

import numpy as np
import pytest

from motor_pipeline.config import load_config
from motor_pipeline.features import extract_gait
from motor_pipeline.io import normalize_imu
from motor_pipeline.synthetic import add_artifact, resting_noise, walking


def extract(frame, cfg=None, placement="ankle"):
    cfg = load_config() if cfg is None else cfg
    recording = normalize_imu(frame, cfg, metadata={"placement": placement})
    return extract_gait(recording, cfg)


@pytest.mark.parametrize("fs", [50, 100, 200])
@pytest.mark.parametrize("cadence", [80, 120, 160])
def test_known_ankle_cadence_and_stride_time(fs, cadence):
    cfg = load_config()
    cfg["input"]["sampling_rate_hz"] = fs
    result = extract(walking(sampling_rate_hz=fs, cadence_steps_min=cadence), cfg)
    assert result["status"] == "ok"
    features = result["features"]
    assert features["cadence_steps_min"] == pytest.approx(cadence, abs=2)
    assert features["mean_stride_time_s"] == pytest.approx(120 / cadence, abs=.02)
    assert features["stride_time_cv"] < .03
    assert features["number_of_strides"] > 10


def test_unique_stride_count_not_sum_of_overlapping_windows():
    result = extract(walking(noise_std=0))
    # Sine positive peaks at .25, 1.25, ...; windows span .5–28.5 s.
    # They retain peaks 1.25–28.25: 28 events and 27 complete strides.
    assert result["features"]["number_of_strides"] == 27
    assert len(result["event_times_s"]) == 28
    repeated_counts = sum(row["features"]["number_of_strides"] for row in result["per_window"])
    assert repeated_counts > 27
    assert result["features"]["cadence_steps_min"] == pytest.approx(120, abs=.2)


def test_recovers_variable_stride_timing_without_normalizing_it_away():
    frame = walking(duration_s=40, noise_std=0)
    intervals = np.tile([.85, 1.15], 25)
    boundaries = np.r_[0, np.cumsum(intervals)]
    phase = np.interp(frame.t, boundaries, np.arange(len(boundaries))) * 2 * np.pi
    frame["gx"] = 40 * np.cos(phase)
    frame["ax"] = .5 * np.cos(phase)
    frame["az"] = 9.80665 + np.cos(2 * phase)
    result = extract(frame)
    assert result["status"] == "ok"
    assert result["features"]["cadence_steps_min"] == pytest.approx(120, abs=3)
    assert result["features"]["stride_time_cv"] == pytest.approx(.15, abs=.035)


def test_fixed_rotation_preserves_cadence_and_swing_amplitude():
    frame = walking(noise_std=0)
    # Rotation mapping the fixture x axis onto z exercises the PCA projection.
    rotated = frame.copy()
    for channels in (["ax", "ay", "az"], ["gx", "gy", "gz"]):
        rotated[channels] = frame[channels].to_numpy()[:, [1, 2, 0]]
    first = extract(frame, placement="wrist")
    second = extract(rotated, placement="wrist")
    for key in ("arm_swing_cycles_min", "arm_swing_amplitude_deg"):
        assert second["features"][key] == pytest.approx(first["features"][key], rel=1e-5)


def test_signed_explicit_gyro_axis_and_wrong_axis_rejection():
    cfg = load_config()
    cfg["features"]["gait"]["gyro_axis"] = "-gx"
    result = extract(walking(), cfg)
    assert result["status"] == "ok"
    assert result["features"]["cadence_steps_min"] == pytest.approx(120, abs=1)
    cfg["features"]["gait"]["gyro_axis"] = "gz"
    assert extract(walking(), cfg)["status"] == "retest"


def test_wrist_is_not_silently_treated_as_ankle():
    result = extract(walking(noise_std=0), placement="wrist")
    f = result["features"]
    assert result["status"] == "ok"
    for key in ("cadence_steps_min", "mean_stride_time_s", "stride_time_cv", "number_of_strides", "walking_speed_proxy_m_s"):
        assert f[key] is None
    assert f["arm_swing_cycles_min"] == pytest.approx(60, abs=1)
    # Angular velocity A*sin(2πft) integrates to peak-to-peak angle A/(πf).
    assert f["arm_swing_amplitude_deg"] == pytest.approx(40 / np.pi, rel=.04)
    assert f["arm_swing_amplitude_cv"] < .02


def test_wrist_stride_mapping_requires_explicit_config():
    cfg = load_config()
    cfg["features"]["gait"]["wrist_one_cycle_per_stride"] = True
    result = extract(walking(), cfg, "wrist")
    assert result["features"]["cadence_steps_min"] == pytest.approx(120, abs=1)
    assert result["features"]["number_of_strides"] == 27


def test_speed_requires_stride_length_and_uses_it_explicitly():
    assert extract(walking())["features"]["walking_speed_proxy_m_s"] is None
    cfg = load_config()
    cfg["features"]["gait"]["assumed_stride_length_m"] = 1.2
    result = extract(walking(), cfg)
    assert result["features"]["walking_speed_proxy_m_s"] == pytest.approx(1.2, abs=.02)
    assert any("not measured walking speed" in text for text in result["limitations"])


def test_single_sensor_limitations_and_unimplemented_turns_are_explicit():
    result = extract(walking())
    assert result["features"]["turn_count"] is None
    assert any("left/right asymmetry" in text for text in result["limitations"])
    assert any("Turn counting is unavailable" in text for text in result["limitations"])


@pytest.mark.parametrize("artifact", ["gap", "missing", "clipping", "duplicate", "out_of_order"])
def test_artifacts_retest_without_features(artifact):
    result = extract(add_artifact(walking(), artifact, duration_s=2))
    assert set(result) == {"status", "reasons"}
    assert result["status"] == "retest"


def test_small_missing_interval_not_converted_to_a_stride():
    frame = walking()
    frame.loc[1400:1409, "gx"] = np.nan
    result = extract(frame)
    assert result["status"] == "ok"
    assert result["features"]["number_of_strides"] < 27
    for row in result["per_window"]:
        assert row["start_s"] + 4 <= 14 or row["start_s"] >= 14.1
    assert max(result["cycle_intervals_s"]) < 1.05


@pytest.mark.parametrize("generator", [resting_noise, lambda: walking(gyro_amplitude=0)])
def test_absent_motion_or_absent_gyro_cycles_retests(generator):
    assert extract(generator())["status"] == "retest"


def test_short_recording_retests():
    assert extract(walking(duration_s=10))["status"] == "retest"


def test_fragmented_signal_retests_despite_low_total_missingness():
    frame = walking()
    frame.loc[np.arange(200, 3000, 200), "gx"] = np.nan
    result = extract(frame)
    assert result["status"] == "retest"


def test_missing_gyroscope_retests():
    cfg = load_config()
    cfg["input"]["allow_missing_gyroscope"] = True
    result = extract(walking().drop(columns=["gx", "gy", "gz"]), cfg)
    assert result["status"] == "retest"


def test_unsupported_placement_retests():
    assert extract(walking(), placement="chest")["status"] == "retest"


def test_metadata_placement_wins_over_config_fallback():
    cfg = load_config()
    cfg["features"]["gait"]["placement"] = "ankle"
    result = extract(walking(), cfg, "wrist")
    assert result["placement"] == "wrist"
    assert result["features"]["cadence_steps_min"] is None


def test_json_safe_spread_and_no_input_mutation():
    recording = normalize_imu(walking(), metadata={"placement": "ankle"})
    before = recording.data.copy(deep=True)
    result = extract_gait(recording)
    assert recording.data.equals(before)
    json.dumps(result, allow_nan=False)
    assert set(result["features"]) == set(result["spread"])
    spread = result["spread"]["cadence_steps_min"]
    assert spread["iqr"][0] <= spread["iqr"][1]
    assert spread["n_valid_windows"] == result["n_valid_windows"]
    assert spread["quality_score"] == result["quality"]["score"]


@pytest.mark.parametrize("key,value", [("gyro_axis", "ax"), ("minimum_peak_distance_s", 0),
                                       ("assumed_stride_length_m", -1), ("wrist_one_cycle_per_stride", 1),
                                       ("minimum_valid_window_fraction", 2), ("minimum_autocorrelation", -1),
                                       ("stride_time_bounds_s", [2, .4]), ("event_merge_tolerance_s", .5)])
def test_invalid_config_fails_explicitly(key, value):
    cfg = load_config()
    cfg["features"]["gait"][key] = value
    with pytest.raises(ValueError):
        extract(walking(), cfg)
