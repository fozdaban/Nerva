"""Load external IMU data into seconds, m/s² and degrees/s.

Repair timestamps only after auditing their original order. Never fill sensor
values or bridge gaps here: downstream quality checks need that evidence.
"""

from copy import deepcopy
from dataclasses import dataclass
from datetime import datetime
import json
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd

from motor_pipeline.config import load_config

AXES = ("ax", "ay", "az", "gx", "gy", "gz")
COLUMNS = ("t", *AXES)


class InputError(ValueError):
    """Malformed recording or unsupported input configuration."""


@dataclass
class Recording:
    """Canonical data and provenance; ingestion does not imply usable quality."""

    data: pd.DataFrame
    metadata: dict[str, Any]
    diagnostics: dict[str, Any]


def _seconds(values: pd.Series, settings: Mapping[str, Any]) -> np.ndarray:
    try:
        if settings["timestamp_format"] == "iso8601":
            # Explicit offsets avoid silently interpreting local wall-clock time.
            parsed = [datetime.fromisoformat(str(v).replace("Z", "+00:00")) for v in values]
            if any(v.tzinfo is None for v in parsed):
                raise InputError("ISO timestamps must include a timezone offset")
            return np.array([v.timestamp() for v in parsed], dtype=float)
        if settings["timestamp_format"] != "numeric":
            raise InputError("timestamp_format must be numeric or iso8601")
        scale = {"s": 1.0, "ms": 1e-3, "us": 1e-6, "ns": 1e-9}
        if settings["timestamp_unit"] not in scale:
            raise InputError("timestamp_unit must be s, ms, us or ns")
        return pd.to_numeric(values, errors="raise").to_numpy(dtype=float) * scale[settings["timestamp_unit"]]
    except (TypeError, ValueError, OverflowError) as exc:
        raise InputError(f"Invalid timestamps: {exc}") from exc


def _settings(config: Mapping[str, Any]) -> tuple[dict[str, Any], float]:
    settings = deepcopy(config["input"])
    fs = float(settings["sampling_rate_hz"])
    low, high = settings["supported_sampling_rate_hz"]
    if not np.isfinite(fs) or not low <= fs <= high:
        raise InputError(f"sampling_rate_hz must be between {low} and {high}")
    if settings["timestamp_mode"] not in ("sample", "packet"):
        raise InputError("timestamp_mode must be sample or packet")
    if settings["packet_timestamp_anchor"] not in ("first", "last"):
        raise InputError("packet_timestamp_anchor must be first or last")
    if settings["timestamp_repair"] not in ("reject", "sort_deduplicate"):
        raise InputError("timestamp_repair must be reject or sort_deduplicate")
    return settings, fs


def normalize_imu(
    frame: pd.DataFrame,
    config: Mapping[str, Any] | None = None,
    metadata: Mapping[str, Any] | None = None,
    *,
    _timestamps_in_seconds: bool = False,
) -> Recording:
    """Normalize a frame; duplicate timestamps keep the first acquired sample.

    For packet-mode CSV, consecutive equal timestamps identify a packet. JSON
    packets preserve explicit packet boundaries in ``load_imu`` instead.
    """
    config = load_config() if config is None else config
    settings, fs = _settings(config)
    if frame.empty:
        raise InputError("Recording contains no samples")
    names = settings["columns"]
    if names["t"] not in frame:
        raise InputError(f"Missing timestamp column: {names['t']}")
    t = (pd.to_numeric(frame[names["t"]], errors="raise").to_numpy(dtype=float)
         if _timestamps_in_seconds else _seconds(frame[names["t"]], settings))
    if not np.isfinite(t).all():
        raise InputError("Timestamps must be finite; missing times cannot be inferred safely")
    packet_count = None
    if settings["timestamp_mode"] == "packet" and not _timestamps_in_seconds:
        starts = np.r_[0, np.flatnonzero(np.diff(t) != 0) + 1]
        ends = np.r_[starts[1:], len(t)]
        packet_count = len(starts)
        for start, end in zip(starts, ends):
            offsets = np.arange(end - start, dtype=float) / fs
            if settings["packet_timestamp_anchor"] == "last":
                offsets -= offsets[-1]
            t[start:end] = t[start] + offsets
    raw = pd.DataFrame({"t": t})
    absent = []
    for axis in AXES:
        source = names[axis]
        if source not in frame:
            if axis.startswith("g") and settings["allow_missing_gyroscope"]:
                raw[axis] = np.nan
                absent.append(axis)
                continue
            raise InputError(f"Missing sensor column: {source}")
        try:
            raw[axis] = pd.to_numeric(frame[source], errors="raise").to_numpy(dtype=float)
        except (TypeError, ValueError) as exc:
            raise InputError(f"Sensor column {source} contains non-numeric values") from exc
    acceleration_scales = {"m/s2": 1.0, "g": settings["standard_gravity_m_s2"]}
    gyro_scales = {"deg/s": 1.0, "rad/s": 180 / np.pi}
    if settings["acceleration_unit"] not in acceleration_scales or settings["gyroscope_unit"] not in gyro_scales:
        raise InputError("Supported units: acceleration m/s2 or g; gyroscope deg/s or rad/s")
    raw.loc[:, ["ax", "ay", "az"]] *= acceleration_scales[settings["acceleration_unit"]]
    raw.loc[:, ["gx", "gy", "gz"]] *= gyro_scales[settings["gyroscope_unit"]]
    mapped = raw.copy()
    axis_map = settings["axis_map"]
    for group in (AXES[:3], AXES[3:]):
        sources = [axis_map[a].removeprefix("-") for a in group]
        if set(sources) != set(group):
            raise InputError("axis_map must be a signed permutation within each sensor")
        for target in group:
            spec = axis_map[target]
            mapped[target] = raw[spec.removeprefix("-")] * (-1 if spec.startswith("-") else 1)
    duplicates = int(mapped.t.duplicated().sum())
    out_of_order = int(np.count_nonzero(np.diff(t) < 0))
    if (duplicates or out_of_order) and settings["timestamp_repair"] == "reject":
        raise InputError(f"Timestamp order invalid: {duplicates} duplicates, {out_of_order} backward transitions")
    cleaned = mapped.sort_values("t", kind="stable").drop_duplicates("t", keep="first").reset_index(drop=True)
    # Subtract after sorting, so all accepted recordings start at zero.
    time_origin = float(cleaned.t.iloc[0])
    cleaned["t"] -= time_origin
    steps = np.diff(cleaned.t.to_numpy())
    gap_mask = steps > float(config["quality"]["gap_periods"]) / fs + 1e-9
    gap_indices = np.flatnonzero(gap_mask)
    missing_equivalent = float(np.maximum(steps * fs - 1, 0)[gap_mask].sum())
    expected = len(cleaned) + missing_equivalent
    values = cleaned[list(AXES)].to_numpy()
    invalid = ~np.isfinite(values)
    cleaned.loc[:, list(AXES)] = np.where(invalid, np.nan, values)
    diagnostics = {
        "input_samples": len(frame), "output_samples": len(cleaned),
        "duplicate_timestamps": duplicates, "out_of_order_transitions": out_of_order,
        "timestamp_repair_applied": bool(duplicates or out_of_order),
        "packet_count": packet_count, "time_origin_seconds": time_origin,
        "gaps": [{"after_sample": int(i), "duration_s": float(steps[i])} for i in gap_indices],
        "estimated_missing_samples": missing_equivalent,
        "estimated_missing_time_fraction": missing_equivalent / expected,
        "nonfinite_sensor_values": int(invalid.sum()),
        "nonfinite_sensor_fraction": float(invalid.mean()),
        "missing_channels": absent,
    }
    meta = {**config["metadata_defaults"], **(metadata or {})}
    # Canonical units and rate are adapter-owned, never caller overrides.
    meta.update({"sampling_rate_hz": fs,
                 "observed_sampling_rate_hz": float(1 / np.median(steps)) if len(steps) else None,
                 "acceleration_unit": "m/s2", "gyroscope_unit": "deg/s"})
    return Recording(cleaned[list(COLUMNS)], meta, diagnostics)


def load_imu(
    path: str | Path,
    config: Mapping[str, Any] | None = None,
    metadata: Mapping[str, Any] | None = None,
    *,
    profile: str = "standard",
) -> Recording:
    """Load CSV, JSON sample rows, or JSON timestamped packets.

    JSON object: {"metadata": {...}, "samples": [...]} or
    {"metadata": {...}, "packets": [{"timestamp": ..., "samples": [...]}]}.
    Each sample is an object using configured source column names.
    """
    config = deepcopy(load_config() if config is None else config)
    if profile == "phyphox":
        config["input"].update(config["input"]["phyphox"])
    elif profile != "standard":
        raise InputError("profile must be standard or phyphox")
    settings, fs = _settings(config)
    path = Path(path)
    file_format = settings["file_format"]
    if file_format == "auto":
        file_format = path.suffix.lower().lstrip(".")
    embedded_metadata: dict[str, Any] = {}
    expanded_packets = False
    packet_count = None
    if file_format == "csv":
        frame = pd.read_csv(path, sep=settings["csv_delimiter"])
    elif file_format == "json":
        with path.open(encoding="utf-8") as handle:
            payload = json.load(handle)
        if isinstance(payload, list):
            frame = pd.DataFrame(payload)
        elif isinstance(payload, dict):
            embedded_metadata = payload.get("metadata", {})
            if not isinstance(embedded_metadata, dict):
                raise InputError("JSON metadata must be an object")
            if ("packets" in payload) == ("samples" in payload):
                raise InputError("JSON object needs exactly one of samples or packets")
            if "packets" in payload:
                rows = []
                packets = payload["packets"]
                if not isinstance(packets, list):
                    raise InputError("packets must be a list")
                packet_count = len(packets)
                for packet in packets:
                    if not isinstance(packet, dict) or not packet.get("samples") or "timestamp" not in packet:
                        raise InputError("Each packet needs a timestamp and nonempty samples")
                    samples = packet["samples"]
                    start = _seconds(pd.Series([packet["timestamp"]]), settings)[0]
                    if settings["packet_timestamp_anchor"] == "last":
                        start -= (len(samples) - 1) / fs
                    for i, sample in enumerate(samples):
                        if not isinstance(sample, dict):
                            raise InputError("Each packet sample must be an object")
                        rows.append({**sample, settings["columns"]["t"]: start + i / fs})
                frame = pd.DataFrame(rows)
                expanded_packets = True
            else:
                frame = pd.DataFrame(payload["samples"])
        else:
            raise InputError("JSON must contain sample rows or an object")
    else:
        raise InputError(f"Unsupported file format: {file_format}")
    result = normalize_imu(frame, config, {**embedded_metadata, **(metadata or {})},
                           _timestamps_in_seconds=expanded_packets)
    if expanded_packets:
        result.diagnostics["packet_count"] = packet_count
    result.metadata["source_format"] = file_format
    result.metadata["input_profile"] = profile
    return result
