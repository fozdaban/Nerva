"""Preprocessing must preserve signals, suppress aliases and isolate gaps."""

import numpy as np
import pytest

from motor_pipeline.config import load_config
from motor_pipeline.io import normalize_imu
from motor_pipeline.preprocess import prepare_imu
from motor_pipeline.synthetic import tremor


def prepare(frame, cfg=None):
    cfg = load_config() if cfg is None else cfg
    return prepare_imu(normalize_imu(frame, cfg), cfg, cfg["features"]["tremor"]["filter_band_hz"])


@pytest.mark.parametrize("source_fs,target_fs", [(50, 100), (100, 100), (200, 100), (200, 50)])
def test_resampling_duration_frequency_and_rms(source_fs, target_fs):
    cfg = load_config()
    cfg["input"]["sampling_rate_hz"] = source_fs
    cfg["preprocess"]["resample_hz"] = target_fs
    result = prepare(tremor(sampling_rate_hz=source_fs, noise_std=0), cfg)
    assert result.dynamic.shape == result.bandpassed.shape
    assert result.dynamic.shape[1:] == (4 * target_fs, 6)
    assert len(result.starts_s) >= 7
    assert result.starts_s[1] - result.starts_s[0] == pytest.approx(2)
    assert np.sqrt(np.mean(result.bandpassed[:, :, 3] ** 2)) == pytest.approx(10 / np.sqrt(2), rel=.03)
    assert np.abs(result.dynamic[:, :, 2]).max() < 1e-5  # constant gravity removed


def test_antialias_before_downsampling():
    cfg = load_config()
    cfg["input"]["sampling_rate_hz"] = 200
    frame = tremor(sampling_rate_hz=200, frequency_hz=90, noise_std=0)
    result = prepare(frame, cfg)
    # Without the resampling anti-alias filter, 90 Hz aliases to a 10 Hz peak.
    assert np.sqrt(np.mean(result.dynamic[:, :, 3] ** 2)) < .05


def test_long_gaps_and_nonfinite_values_do_not_cross_windows():
    cfg = load_config()
    frame = tremor(duration_s=30)
    frame.loc[1400:1499, "gx"] = np.nan
    result = prepare(frame, cfg)
    assert result.starts_s
    assert np.isfinite(result.dynamic).all()
    for start in result.starts_s:
        assert start + 4 <= 14 or start >= 15


def test_no_interpolation_across_removed_samples_beyond_limit():
    frame = tremor(duration_s=30).drop(index=range(1400, 1410)).reset_index(drop=True)
    result = prepare(frame)
    for start in result.starts_s:
        assert start + 4 <= 14 or start >= 14.1


def test_clipped_samples_are_excluded_without_modifying_input():
    frame = tremor(duration_s=30)
    frame.loc[1400:1409, "gx"] = 1000
    snapshot = frame.copy(deep=True)
    result = prepare(frame)
    assert frame.equals(snapshot)
    for start in result.starts_s:
        assert start + 4 <= 14 or start >= 14.1


def test_all_missing_returns_empty_windows():
    frame = tremor()
    frame.loc[:, "gx"] = np.nan
    result = prepare(frame)
    assert result.dynamic.shape == (0, 400, 6)


def test_higher_order_filter_remains_finite():
    cfg = load_config()
    cfg["preprocess"]["filter_order"] = 8
    result = prepare(tremor(), cfg)
    assert np.isfinite(result.dynamic).all()
    assert np.isfinite(result.bandpassed).all()
    assert np.sqrt(np.mean(result.bandpassed[:, :, 3] ** 2)) == pytest.approx(10 / np.sqrt(2), rel=.05)


@pytest.mark.parametrize("key,value", [("window_s", 0), ("overlap_fraction", 1),
                                       ("filter_order", 0), ("resample_hz", 20),
                                       ("edge_trim_s", -1), ("maximum_interpolation_gap_s", 0)])
def test_invalid_config(key, value):
    cfg = load_config()
    cfg["preprocess"][key] = value
    with pytest.raises(ValueError):
        prepare(tremor(), cfg)
