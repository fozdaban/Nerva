"""Task quality gates must fail safely before producing motor metrics."""

from copy import deepcopy
import json

import numpy as np
import pytest

from motor_pipeline.config import load_config
from motor_pipeline.io import Recording, normalize_imu
from motor_pipeline.quality import assess_imu
from motor_pipeline.synthetic import add_artifact, resting_noise, tapping, tremor, walking


def assess(frame, task="tremor", config=None):
    cfg = load_config() if config is None else config
    return assess_imu(normalize_imu(frame, cfg), task, cfg)


@pytest.mark.parametrize("fs", [50, 100, 200])
@pytest.mark.parametrize("generator,task", [(tremor, "tremor"), (walking, "gait"), (tapping, "tapping_imu")])
def test_clean_signals_pass(fs, generator, task):
    cfg = load_config()
    cfg["input"]["sampling_rate_hz"] = fs
    report = assess(generator(sampling_rate_hz=fs), task, cfg)
    assert report.status == "ok"
    assert report.score == pytest.approx(1)
    assert report.reasons == []
    assert report.retest_response() is None


@pytest.mark.parametrize("task", ["tremor", "rest_tremor", "postural_tremor", "sway"])
def test_stationary_tasks_do_not_require_movement(task):
    report = assess(resting_noise(accel_noise_std=0, gyro_noise_std=0), task)
    assert report.status == "ok"


@pytest.mark.parametrize("task", ["gait", "tapping_imu", "reps"])
def test_movement_required_for_active_tasks(task):
    report = assess(resting_noise(accel_noise_std=0, gyro_noise_std=0), task)
    assert report.status == "retest"
    assert any("movement" in reason for reason in report.reasons)
    assert report.measurements["movement_rms_m_s2"] == pytest.approx(0, abs=1e-10)


@pytest.mark.parametrize("artifact,reason", [("gap", "gap"), ("clipping", "clipping"),
                                           ("missing", "Missing"), ("duplicate", "Duplicate"),
                                           ("out_of_order", "Out-of-order")])
def test_artifacts_retest_without_metrics(artifact, reason):
    report = assess(add_artifact(tremor(), artifact, duration_s=2))
    assert report.status == "retest"
    assert report.score < .75
    assert any(reason in r for r in report.reasons)
    response = report.retest_response()
    assert set(response) == {"status", "reasons"}
    json.dumps(response, allow_nan=False)
    json.dumps(report.as_dict(), allow_nan=False)
    json.dumps(report.measurements, allow_nan=False)


def test_gap_rejected_even_when_total_loss_is_small():
    report = assess(add_artifact(tremor(duration_s=100), "gap", duration_s=.5))
    assert report.measurements["missing_fraction"] == pytest.approx(.005)
    assert any("gap" in reason for reason in report.reasons)


def test_single_dropped_slots_count_toward_missingness():
    frame = tremor().drop(index=np.arange(10, 1990, 10)).reset_index(drop=True)
    report = assess(frame)
    assert report.measurements["gap_count"] == 0
    assert report.measurements["estimated_missing_samples"] == 198
    assert report.measurements["missing_fraction"] == pytest.approx(.099)
    assert report.status == "retest"


def test_missing_union_does_not_average_away_required_channel_loss():
    frame = tremor()
    frame.loc[:199, "gx"] = np.nan
    frame.loc[100:299, "ay"] = np.nan
    report = assess(frame)
    assert report.measurements["missing_fraction"] == pytest.approx(.15)
    assert report.status == "retest"


def test_gaps_and_sensor_missingness_do_not_double_count():
    frame = add_artifact(tremor(), "gap", duration_s=.1)
    frame.loc[:9, "ax"] = np.nan
    report = assess(frame)
    assert report.measurements["missing_fraction"] == pytest.approx(.01)
    assert report.status == "ok"


@pytest.mark.parametrize("task,expected", [("tremor", "retest"), ("sway", "ok")])
def test_required_channels_are_task_specific(task, expected):
    cfg = load_config()
    cfg["input"]["allow_missing_gyroscope"] = True
    frame = resting_noise().drop(columns=["gx", "gy", "gz"])
    report = assess(frame, task, cfg)
    assert report.status == expected


@pytest.mark.parametrize("duration,status", [(9.99, "retest"), (10, "ok")])
def test_duration_boundary(duration, status):
    report = assess(tremor(duration_s=duration))
    assert report.status == status
    assert report.measurements["duration_s"] == pytest.approx(duration)


def test_implausible_values_are_not_a_motor_metric():
    frame = walking()
    frame.loc[0, "ax"] = 1e308
    with np.errstate(over="raise", invalid="raise"):
        report = assess(frame, "gait")
    assert report.status == "retest"
    assert any("Implausible" in reason for reason in report.reasons)
    json.dumps(report.measurements, allow_nan=False)


def test_negative_gyro_clipping_uses_configured_range():
    cfg = load_config()
    cfg["input"]["gyroscope_range_deg_s"] = 100
    frame = tremor()
    frame.loc[:39, "gz"] = -99
    report = assess(frame, config=cfg)
    assert report.measurements["clipped_fraction"] == pytest.approx(.02)
    assert report.status == "retest"


def test_small_loss_is_retained_in_score_without_retest():
    frame = tremor()
    frame.loc[:19, "gx"] = np.nan
    report = assess(frame)
    assert report.status == "ok"
    assert report.score == pytest.approx(.99)


def test_threshold_changes_and_custom_task_mapping():
    cfg = load_config()
    cfg["task_mapping"]["my_test"] = "tremor"
    frame = add_artifact(tremor(), "clipping", duration_s=.4)
    assert assess(frame, "my_test", cfg).status == "retest"
    cfg["quality"]["maximum_clipped_fraction"] = .03
    assert assess(frame, "my_test", cfg).status == "ok"


def test_unknown_task_fails_explicitly():
    with pytest.raises(ValueError, match="Unknown IMU task"):
        assess(tremor(), "typo")


def test_metadata_task_and_input_immutability():
    cfg = load_config()
    recording = normalize_imu(tremor(), cfg, {"task_name": "rest_tremor"})
    snapshot = recording.data.copy(deep=True)
    original_cfg = deepcopy(cfg)
    assert assess_imu(recording, config=cfg).status == "ok"
    assert recording.data.equals(snapshot)
    assert cfg == original_cfg


def test_sampling_rate_disagreement_retests():
    frame = tremor(sampling_rate_hz=50)
    report = assess(frame)  # Configured at 100 Hz deliberately.
    assert report.status == "retest"
    assert any("sampling rate" in reason for reason in report.reasons)


@pytest.mark.parametrize("problem", ["empty", "one", "backward", "nan_time", "no_ax", "no_t", "zero_rate"])
def test_missing_or_broken_canonical_recordings_retest(problem):
    recording = normalize_imu(tremor())
    if problem == "empty":
        recording.data = recording.data.iloc[:0]
    elif problem == "one":
        recording.data = recording.data.iloc[:1]
    elif problem == "backward":
        recording.data.loc[3, "t"] = 0
    elif problem == "nan_time":
        recording.data.loc[3, "t"] = np.nan
    elif problem == "no_ax":
        recording.data = recording.data.drop(columns="ax")
    elif problem == "no_t":
        recording.data = recording.data.drop(columns="t")
    else:
        recording.metadata["sampling_rate_hz"] = 0
    report = assess_imu(recording, "tremor")
    assert report.status == "retest"
    assert report.score == 0


@pytest.mark.parametrize("key,value", [("minimum_score", 0), ("minimum_score", 1.2),
                                       ("maximum_missing_fraction", -1), ("maximum_gap_s", 0),
                                       ("movement_rms_min_m_s2", float("nan")),
                                       ("implausible_range_multiplier", .5)])
def test_invalid_quality_config(key, value):
    cfg = load_config()
    cfg["quality"][key] = value
    with pytest.raises(ValueError):
        assess(tremor(), config=cfg)


def test_strict_overall_threshold_can_trigger_retest():
    cfg = load_config()
    cfg["quality"]["minimum_score"] = .999
    frame = tremor()
    frame.loc[:19, "gx"] = np.nan
    report = assess(frame, config=cfg)
    assert report.status == "retest"
    assert report.score == pytest.approx(.99)
    assert "Combined" in report.reasons[0]
