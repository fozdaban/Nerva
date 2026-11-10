"""Known acceleration trajectories and conservative sway input handling."""

import json

import numpy as np
import pytest

from motor_pipeline.config import load_config
from motor_pipeline.features.sway import extract_sway
from motor_pipeline.io import normalize_imu
from motor_pipeline.synthetic import add_artifact, resting_noise


def config():
    cfg = load_config()
    cfg["features"]["sway"].update(supported_placements=["lower_back", "chest"],
                                   edge_trim_s=3.0, minimum_valid_window_fraction=.5,
                                   require_horizontal_alignment_confirmation=True)
    return cfg


def ellipse(duration_s=40, fs=100, a=.2, b=.1, frequency=.5):
    frame = resting_noise(duration_s=duration_s, sampling_rate_hz=fs,
                          accel_noise_std=0, gyro_noise_std=0)
    frame["ax"] = a * np.sin(2 * np.pi * frequency * frame.t)
    frame["ay"] = b * np.cos(2 * np.pi * frequency * frame.t)
    return frame


def extract(frame, cfg=None, placement="lower_back", aligned=True):
    cfg = config() if cfg is None else cfg
    recording = normalize_imu(frame, cfg, {"placement": placement, "horizontal_alignment_confirmed": aligned})
    return extract_sway(recording, cfg)


@pytest.mark.parametrize("fs", [50, 100, 200])
def test_known_ellipse_rms_frequency_area(fs):
    cfg = config()
    cfg["input"]["sampling_rate_hz"] = fs
    result = extract(ellipse(fs=fs), cfg)
    assert result["status"] == "ok"
    f = result["features"]
    assert f["horizontal_acceleration_rms_m_s2"] == pytest.approx(np.sqrt((.2 ** 2 + .1 ** 2) / 2), rel=.06)
    assert f["mean_frequency_hz"] == pytest.approx(.5, abs=.04)
    # Nominal Gaussian ellipse uses chi-square(2,.95) = -2*ln(.05).
    expected = np.pi * (-2 * np.log(.05)) * (.2 * .1 / 2)
    assert f["acceleration_ellipse_area_95_m2_s4"] == pytest.approx(expected, rel=.1)


def test_circular_acceleration_path_has_known_circumference():
    result = extract(ellipse(a=.2, b=.2))
    # Four-second window with 0.5 Hz motion contains ~2 circles.
    assert result["per_window"][0]["features"]["acceleration_path_length_proxy_m_s2"] == pytest.approx(2 * 2 * np.pi * .2, rel=.06)
    expected = result["observed_duration_s"] * .5 * 2 * np.pi * .2
    assert result["features"]["acceleration_path_length_proxy_m_s2"] == pytest.approx(expected, rel=.06)
    repeated = sum(row["features"]["acceleration_path_length_proxy_m_s2"] for row in result["per_window"])
    assert repeated > 1.5 * result["features"]["acceleration_path_length_proxy_m_s2"]


def test_plane_rotation_preserves_vector_metrics():
    frame = ellipse()
    rotated = frame.copy()
    angle = .7
    rotation = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
    rotated[["ax", "ay"]] = frame[["ax", "ay"]].to_numpy() @ rotation.T
    first, second = extract(frame), extract(rotated)
    for name in first["features"]:
        assert first["features"][name] == pytest.approx(second["features"][name], rel=1e-8)


def test_acceleration_only_recording_is_supported():
    cfg = config()
    cfg["input"]["allow_missing_gyroscope"] = True
    result = extract(ellipse().drop(columns=["gx", "gy", "gz"]), cfg)
    assert result["status"] == "ok"
    assert result["features"]["mean_frequency_hz"] == pytest.approx(.5, abs=.04)


def test_static_recording_remains_valid_with_no_frequency():
    result = extract(resting_noise(duration_s=40, accel_noise_std=0, gyro_noise_std=0))
    assert result["status"] == "ok"
    assert result["features"]["horizontal_acceleration_rms_m_s2"] == 0
    assert result["features"]["mean_frequency_hz"] is None
    assert result["features"]["acceleration_ellipse_area_95_m2_s4"] == 0


def test_degenerate_line_has_zero_ellipse_area():
    result = extract(ellipse(b=0))
    assert result["features"]["acceleration_ellipse_area_95_m2_s4"] == pytest.approx(0, abs=1e-12)


def test_configurable_horizontal_axes_and_chest_placement():
    cfg = config()
    cfg["features"]["sway"]["horizontal_axes"] = ["ay", "az"]
    frame = ellipse()
    frame[["ay", "az"]] = frame[["ax", "ay"]].to_numpy()
    frame["ax"] = 9.80665
    result = extract(frame, cfg, placement="chest")
    assert result["status"] == "ok"
    assert result["horizontal_axes"] == ["ay", "az"]
    assert result["features"]["mean_frequency_hz"] == pytest.approx(.5, abs=.04)


@pytest.mark.parametrize("placement,aligned", [("wrist", True), ("ankle", True), ("lower_back", False)])
def test_wrong_placement_or_unconfirmed_alignment_retests(placement, aligned):
    result = extract(ellipse(), placement=placement, aligned=aligned)
    assert result["status"] == "retest"
    assert set(result) == {"status", "reasons"}


def test_alignment_requirement_can_be_explicitly_relaxed():
    cfg = config()
    cfg["features"]["sway"]["require_horizontal_alignment_confirmation"] = False
    assert extract(ellipse(), cfg, aligned=False)["status"] == "ok"


@pytest.mark.parametrize("artifact", ["gap", "clipping", "missing", "duplicate", "out_of_order"])
def test_poor_quality_retests_without_features(artifact):
    result = extract(add_artifact(ellipse(), artifact, duration_s=3))
    assert result["status"] == "retest"
    assert "features" not in result


def test_sparse_missing_data_excludes_affected_windows():
    frame = ellipse(duration_s=50)
    frame.loc[2400:2409, "ax"] = np.nan
    result = extract(frame)
    assert result["status"] == "ok"
    assert result["n_valid_windows"] < result["expected_windows"]
    for row in result["per_window"]:
        assert row["start_s"] + 4 <= 24 or row["start_s"] >= 24.1
    assert result["observed_duration_s"] < 44


def test_fragmentation_and_short_input_retest():
    frame = ellipse()
    frame.loc[np.arange(300, 4000, 300), "ax"] = np.nan
    assert extract(frame)["status"] == "retest"
    assert extract(ellipse(duration_s=10))["status"] == "retest"


def test_json_safe_spread_and_no_input_mutation():
    cfg = config()
    record = normalize_imu(ellipse(), cfg, {"placement": "lower_back", "horizontal_alignment_confirmed": True})
    original = record.data.copy(deep=True)
    result = extract_sway(record, cfg)
    assert record.data.equals(original)
    json.dumps(result, allow_nan=False)
    assert set(result["features"]) == set(result["spread"])
    assert result["spread"]["horizontal_acceleration_rms_m_s2"]["n_valid_windows"] == result["n_valid_windows"]
    assert any("not body displacement" in x for x in result["limitations"])


@pytest.mark.parametrize("key,value", [("horizontal_axes", ["ax", "ax"]), ("filter_band_hz", [.1, 100]),
                                       ("edge_trim_s", 0), ("minimum_valid_window_fraction", 2),
                                       ("require_horizontal_alignment_confirmation", "yes")])
def test_invalid_config_raises(key, value):
    cfg = config()
    cfg["features"]["sway"][key] = value
    with pytest.raises(ValueError):
        extract(ellipse(), cfg)
