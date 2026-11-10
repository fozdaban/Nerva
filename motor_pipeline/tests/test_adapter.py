import json

import numpy as np
import pandas as pd
import pytest

from motor_pipeline.config import load_config
from motor_pipeline.io import InputError, load_imu, normalize_imu


def samples(n=100, fs=100):
    return pd.DataFrame({"t": 10 + np.arange(n) / fs, "ax": np.ones(n),
                         "ay": np.zeros(n), "az": np.full(n, 9.80665),
                         "gx": np.ones(n), "gy": np.zeros(n), "gz": np.zeros(n)})


@pytest.mark.parametrize("fs", [50, 100, 200])
def test_canonical_format(fs):
    cfg = load_config()
    cfg["input"]["sampling_rate_hz"] = fs
    result = normalize_imu(samples(fs=fs), cfg)
    assert list(result.data) == ["t", "ax", "ay", "az", "gx", "gy", "gz"]
    assert result.data.t.iloc[0] == 0
    assert all(dtype == np.dtype("float64") for dtype in result.data.dtypes)
    assert result.metadata["observed_sampling_rate_hz"] == pytest.approx(fs)
    assert result.diagnostics["gaps"] == []


@pytest.mark.parametrize("unit,multiplier", [("s", 1), ("ms", 1e3), ("us", 1e6), ("ns", 1e9)])
def test_numeric_timestamp_units(unit, multiplier):
    cfg = load_config()
    cfg["input"]["timestamp_unit"] = unit
    frame = samples()
    frame.t *= multiplier
    np.testing.assert_allclose(normalize_imu(frame, cfg).data.t, np.arange(100) / 100, atol=1e-12)


def test_iso_times_and_missing_timezone():
    cfg = load_config()
    cfg["input"]["timestamp_format"] = "iso8601"
    frame = samples(2)
    frame["t"] = ["2026-01-01T00:00:00Z", "2026-01-01T00:00:00.010+00:00"]
    assert normalize_imu(frame, cfg).data.t.iloc[1] == pytest.approx(0.01, abs=1e-6)
    frame["t"] = ["2026-01-01T00:00:00", "2026-01-01T00:00:00.010"]
    with pytest.raises(InputError, match="timezone"):
        normalize_imu(frame, cfg)


def test_units_and_signed_axis_permutation():
    cfg = load_config()
    cfg["input"].update(acceleration_unit="g", gyroscope_unit="rad/s")
    cfg["input"]["axis_map"].update(ax="-ay", ay="ax")
    result = normalize_imu(samples(), cfg)
    assert result.data.ay.iloc[0] == pytest.approx(9.80665)
    assert result.data.ax.iloc[0] == 0
    assert result.data.gx.iloc[0] == pytest.approx(180 / np.pi)


def test_audit_precedes_repair_and_preserves_first_duplicate():
    frame = samples(4)
    frame = pd.concat([frame.iloc[[2]], frame.iloc[[0]], frame.iloc[[1]], frame.iloc[[2]], frame.iloc[[3]]])
    frame.iloc[3, frame.columns.get_loc("ax")] = 99
    result = normalize_imu(frame)
    assert result.diagnostics["duplicate_timestamps"] == 1
    assert result.diagnostics["out_of_order_transitions"] == 1
    assert result.diagnostics["timestamp_repair_applied"]
    assert len(result.data) == 4
    assert result.data.ax.max() == 1
    cfg = load_config()
    cfg["input"]["timestamp_repair"] = "reject"
    with pytest.raises(InputError, match="Timestamp order invalid"):
        normalize_imu(frame, cfg)


def test_gaps_and_nonfinite_values_remain_visible():
    frame = samples(100).drop(index=range(20, 30)).reset_index(drop=True)
    frame.loc[0, "ax"] = np.inf
    frame.loc[1, "gy"] = np.nan
    result = normalize_imu(frame)
    assert len(result.diagnostics["gaps"]) == 1
    assert result.diagnostics["gaps"][0]["duration_s"] == pytest.approx(0.11)
    assert result.diagnostics["estimated_missing_samples"] == pytest.approx(10)
    assert result.diagnostics["estimated_missing_time_fraction"] == pytest.approx(0.1)
    assert result.diagnostics["nonfinite_sensor_values"] == 2
    assert result.data.ax.isna().sum() == 1
    assert len(result.data) == 90  # no gap filling


def test_gap_threshold_strictly_greater_than_two_periods():
    frame = samples(10).drop(index=3)
    assert normalize_imu(frame).diagnostics["gaps"] == []


@pytest.mark.parametrize("problem", ["empty", "no_time", "nan_time", "no_accel", "text"])
def test_invalid_data(problem):
    frame = samples()
    if problem == "empty":
        frame = frame.iloc[:0]
    elif problem == "no_time":
        frame = frame.drop(columns="t")
    elif problem == "nan_time":
        frame.loc[0, "t"] = np.nan
    elif problem == "no_accel":
        frame = frame.drop(columns="ax")
    else:
        frame["gx"] = "bad"
    with pytest.raises(InputError):
        normalize_imu(frame)


def test_optional_gyro_is_nan_never_zero():
    frame = samples().drop(columns=["gx", "gy", "gz"])
    with pytest.raises(InputError):
        normalize_imu(frame)
    cfg = load_config()
    cfg["input"]["allow_missing_gyroscope"] = True
    result = normalize_imu(frame, cfg)
    assert result.data[["gx", "gy", "gz"]].isna().all().all()
    assert result.diagnostics["missing_channels"] == ["gx", "gy", "gz"]
    assert result.diagnostics["nonfinite_sensor_fraction"] == 0.5


def test_csv_json_roundtrip_and_metadata(tmp_path):
    frame = samples()
    csv = tmp_path / "imu.csv"
    frame.to_csv(csv, index=False)
    source = tmp_path / "imu.json"
    source.write_text(json.dumps({"metadata": {"device": "test", "side": "left"},
                                  "samples": frame.to_dict("records")}), encoding="utf-8")
    first = load_imu(csv)
    second = load_imu(source, metadata={"side": "right", "sampling_rate_hz": 999})
    np.testing.assert_allclose(first.data, second.data)
    assert second.metadata["device"] == "test"
    assert second.metadata["side"] == "right"
    assert second.metadata["sampling_rate_hz"] == 100
    json.dumps(second.diagnostics, allow_nan=False)
    json.dumps(second.metadata, allow_nan=False)
    source.write_text(frame.to_json(orient="records"), encoding="utf-8")
    assert len(load_imu(source).data) == 100


@pytest.mark.parametrize("anchor", ["first", "last"])
def test_json_packets_gap_and_anchor(tmp_path, anchor):
    cfg = load_config()
    cfg["input"]["packet_timestamp_anchor"] = anchor
    cfg["input"]["timestamp_unit"] = "ms"
    rows = samples(3).drop(columns="t").to_dict("records")
    source = tmp_path / "packets.json"
    source.write_text(json.dumps({"packets": [
        {"timestamp": 10000, "samples": rows}, {"timestamp": 10100, "samples": rows}
    ]}), encoding="utf-8")
    result = load_imu(source, cfg)
    np.testing.assert_allclose(result.data.t, [0, .01, .02, .10, .11, .12], atol=1e-12)
    assert result.diagnostics["packet_count"] == 2
    assert result.diagnostics["estimated_missing_samples"] == pytest.approx(7)
    assert result.diagnostics["time_origin_seconds"] == pytest.approx(10 if anchor == "first" else 9.98)


def test_overlapping_packets_are_audited(tmp_path):
    source = tmp_path / "packets.json"
    rows = samples(3).drop(columns="t").to_dict("records")
    source.write_text(json.dumps({"packets": [
        {"timestamp": 0, "samples": rows}, {"timestamp": 0, "samples": rows}
    ]}), encoding="utf-8")
    result = load_imu(source)
    assert result.diagnostics["duplicate_timestamps"] == 3
    assert result.diagnostics["out_of_order_transitions"] == 1


@pytest.mark.parametrize("anchor", ["first", "last"])
def test_csv_packet_timestamps(anchor):
    cfg = load_config()
    cfg["input"].update(timestamp_mode="packet", packet_timestamp_anchor=anchor)
    frame = samples(6)
    frame["t"] = [10, 10, 10, 10.03, 10.03, 10.03]
    result = normalize_imu(frame, cfg)
    np.testing.assert_allclose(result.data.t, np.arange(6) / 100, atol=1e-12)
    assert result.diagnostics["duplicate_timestamps"] == 0
    assert result.diagnostics["packet_count"] == 2


def test_phyphox_profile_and_header_mapping(tmp_path):
    cfg = load_config()
    names = cfg["input"]["phyphox"]["columns"]
    frame = samples().rename(columns=names)
    path = tmp_path / "phone.csv"
    frame.to_csv(path, index=False)
    result = load_imu(path, cfg, profile="phyphox")
    assert result.data.gx.iloc[0] == pytest.approx(180 / np.pi)
    assert cfg["input"]["columns"]["t"] == "t"  # caller config is not mutated


@pytest.mark.parametrize("setting,value", [("sampling_rate_hz", 20), ("timestamp_unit", "minutes"),
                                           ("timestamp_mode", "guess"), ("timestamp_repair", "guess"),
                                           ("acceleration_unit", "unknown"), ("packet_timestamp_anchor", "middle")])
def test_invalid_config(setting, value):
    cfg = load_config()
    cfg["input"][setting] = value
    with pytest.raises(InputError):
        normalize_imu(samples(), cfg)


def test_invalid_axis_map():
    cfg = load_config()
    cfg["input"]["axis_map"]["ax"] = "gx"
    with pytest.raises(InputError, match="permutation"):
        normalize_imu(samples(), cfg)
