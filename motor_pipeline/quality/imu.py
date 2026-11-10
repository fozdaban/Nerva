"""Conservative engineering checks; scores are not calibrated probabilities."""

from dataclasses import dataclass
from typing import Any, Mapping

import numpy as np

from motor_pipeline.config import load_config
from motor_pipeline.io.adapter import AXES, Recording


@dataclass(frozen=True)
class IMUQuality:
    """Internal audit details plus the agreed score/reasons output fields."""

    score: float
    reasons: list[str]
    measurements: dict[str, Any]
    status: str

    def as_dict(self) -> dict[str, Any]:
        """Quality object for a future task envelope; no output-schema additions."""
        return {"score": self.score, "reasons": list(self.reasons)}

    def retest_response(self) -> dict[str, Any] | None:
        """Return the requested minimal failure response, containing no metrics."""
        if self.status == "retest":
            return {"status": "retest", "reasons": list(self.reasons)}
        return None


def _validate(config: Mapping[str, Any], task: str) -> tuple[Mapping[str, Any], list[str]]:
    q = config["quality"]
    if task not in q["task_requirements"] or task not in q["minimum_duration_s"]:
        raise ValueError(f"Unknown IMU task: {task}; configure task_mapping/task_requirements")
    for key in ("minimum_score", "clipping_range_fraction", "maximum_missing_fraction",
                "maximum_clipped_fraction", "maximum_implausible_fraction",
                "maximum_duplicate_fraction", "maximum_out_of_order_fraction",
                "maximum_rate_relative_error"):
        value = float(q[key])
        if not np.isfinite(value) or not 0 <= value <= 1:
            raise ValueError(f"quality.{key} must be finite and between 0 and 1")
    if q["minimum_score"] == 0 or q["clipping_range_fraction"] == 0:
        raise ValueError("minimum_score and clipping_range_fraction must be positive")
    for key, value in {
        "maximum_gap_s": q["maximum_gap_s"], "gap_periods": q["gap_periods"],
        "movement_rms_min_m_s2": q["movement_rms_min_m_s2"],
        "minimum_duration_s": q["minimum_duration_s"][task],
        "acceleration_range_g": config["input"]["acceleration_range_g"],
        "standard_gravity_m_s2": config["input"]["standard_gravity_m_s2"],
        "gyroscope_range_deg_s": config["input"]["gyroscope_range_deg_s"],
    }.items():
        if not np.isfinite(value) or value <= 0:
            raise ValueError(f"{key} must be finite and positive")
    multiplier = q["implausible_range_multiplier"]
    if not np.isfinite(multiplier) or multiplier < 1:
        raise ValueError("implausible_range_multiplier must be at least 1")
    policy = q["task_requirements"][task]
    channels = policy["required_channels"]
    if not channels or len(set(channels)) != len(channels) or not set(channels) <= set(AXES):
        raise ValueError("required_channels must be unique canonical sensor channels")
    if not isinstance(policy["movement_expected"], bool):
        raise ValueError("movement_expected must be a boolean")
    if policy["movement_expected"] and not {"ax", "ay", "az"} <= set(channels):
        raise ValueError("Movement RMS requires all three acceleration channels")
    return q, list(channels)


def assess_imu(
    recording: Recording,
    task: str | None = None,
    config: Mapping[str, Any] | None = None,
) -> IMUQuality:
    """Assess canonical IMU signals without modifying or imputing them.

    Missingness counts time slots with *any* required channel unavailable, plus
    inferred dropped slots. Stationary tremor/sway tasks may be perfectly valid.
    Unknown tasks/configuration raise ValueError rather than silently passing.
    """
    config = load_config() if config is None else config
    task = task if task is not None else recording.metadata.get("task_name", "unspecified")
    task = config["task_mapping"].get(task, task)
    q, channels = _validate(config, task)
    reasons: list[str] = []
    measures: dict[str, Any] = {"task": task, "required_channels": channels}
    scores = [1.0]

    def fail(reason: str) -> None:
        reasons.append(reason)

    def result() -> IMUQuality:
        score = float(np.clip(min(scores), 0, 1))
        # A failed engineering gate always forces a score below the cutoff.
        if reasons:
            score = min(score, float(q["minimum_score"]) / 2)
        if score < q["minimum_score"] and not reasons:
            fail("Combined data quality is below the configured threshold.")
        return IMUQuality(score, reasons, measures, "retest" if reasons else "ok")

    frame = recording.data
    if len(frame) < 2 or "t" not in frame:
        fail("At least two timestamped samples are required.")
        scores.append(0.0)
        return result()
    absent = [name for name in channels if name not in frame or not np.isfinite(frame[name].to_numpy(dtype=float)).any()]
    if absent:
        fail(f"Required channels are missing or entirely nonfinite: {', '.join(absent)}.")
        scores.append(0.0)
        measures["missing_channels"] = absent
        return result()
    fs = float(recording.metadata.get("sampling_rate_hz", float("nan")))
    t = frame.t.to_numpy(dtype=float)
    dt = np.diff(t)
    if not np.isfinite(fs) or fs <= 0 or not np.isfinite(t).all() or np.any(dt <= 0):
        fail("Canonical timestamps must be finite, strictly increasing, with a positive sampling rate.")
        scores.append(0.0)
        return result()
    low, high = config["input"]["supported_sampling_rate_hz"]
    if not low <= fs <= high:
        fail("Recording sampling rate is outside the configured supported range.")
    # Last timestamp + one period represents the duration of all acquired slots.
    duration = float(t[-1] - t[0] + 1 / fs)
    minimum_duration = float(q["minimum_duration_s"][task])
    measures["duration_s"] = duration
    scores.append(min(1.0, duration / minimum_duration))
    if duration + 1e-9 < minimum_duration:
        fail(f"Recording is too short for {task}: {duration:.3f} s; need {minimum_duration:g} s.")

    values = frame[channels].to_numpy(dtype=float)
    finite = np.isfinite(values)
    missing_rows = int((~finite.all(axis=1)).sum())
    # Also count single dropped slots (2 periods), below the >2-period gap flag.
    lost_slots = float(np.maximum(np.rint(dt * fs) - 1, 0).sum())
    missing_fraction = float((missing_rows + lost_slots) / (len(frame) + lost_slots))
    gap_mask = dt > q["gap_periods"] / fs + 1e-9
    longest_gap = float(dt[gap_mask].max()) if gap_mask.any() else 0.0
    measures.update(missing_fraction=missing_fraction, missing_required_rows=missing_rows,
                    estimated_missing_samples=lost_slots, gap_count=int(gap_mask.sum()),
                    longest_gap_s=longest_gap)
    scores.append(1 - missing_fraction)
    if missing_fraction > q["maximum_missing_fraction"] + 1e-12:
        fail(f"Missing required signal: {missing_fraction:.1%}; limit {q['maximum_missing_fraction']:.1%}.")
    if longest_gap > q["maximum_gap_s"] + 1e-9:
        fail(f"Timestamp gap is too long: {longest_gap:.3f} s; limit {q['maximum_gap_s']:g} s.")

    acceleration_range = config["input"]["acceleration_range_g"] * config["input"]["standard_gravity_m_s2"]
    ranges = np.array([acceleration_range if c.startswith("a") else config["input"]["gyroscope_range_deg_s"] for c in channels])
    clipped_rows = np.any(finite & (np.abs(values) >= ranges * q["clipping_range_fraction"]), axis=1)
    implausible_rows = np.any(finite & (np.abs(values) > ranges * q["implausible_range_multiplier"]), axis=1)
    clipped_fraction = float(clipped_rows.mean())
    implausible_fraction = float(implausible_rows.mean())
    measures.update(clipped_fraction=clipped_fraction, implausible_fraction=implausible_fraction)
    scores.extend([1 - clipped_fraction, 1 - implausible_fraction])
    if clipped_fraction > q["maximum_clipped_fraction"] + 1e-12:
        fail(f"Sensor clipping near full scale: {clipped_fraction:.1%}; limit {q['maximum_clipped_fraction']:.1%}.")
    if implausible_fraction > q["maximum_implausible_fraction"] + 1e-12:
        fail(f"Implausible values exceed the configured sensor range tolerance: {implausible_fraction:.1%}.")

    observed_fs = float(1 / np.median(dt))
    rate_error = abs(observed_fs / fs - 1)
    measures.update(observed_sampling_rate_hz=observed_fs, rate_relative_error=rate_error)
    scores.append(max(0.0, 1 - rate_error))
    if rate_error > q["maximum_rate_relative_error"] + 1e-12:
        fail(f"Observed sampling rate {observed_fs:.2f} Hz differs from configured {fs:g} Hz.")
    audit = recording.diagnostics
    input_count = max(len(frame), int(audit.get("input_samples", len(frame))))
    for name, key, maximum in (
        ("Duplicate timestamps", "duplicate_timestamps", "maximum_duplicate_fraction"),
        ("Out-of-order transitions", "out_of_order_transitions", "maximum_out_of_order_fraction"),
    ):
        fraction = float(audit.get(key, 0) / input_count)
        measures[f"{key}_fraction"] = fraction
        scores.append(max(0.0, 1 - fraction))
        if fraction > q[maximum] + 1e-12:
            fail(f"{name} before adapter repair: {fraction:.1%}; limit {q[maximum]:.1%}.")

    if q["task_requirements"][task]["movement_expected"] and not implausible_rows.any():
        acceleration = frame[["ax", "ay", "az"]].to_numpy(dtype=float)
        complete = acceleration[np.isfinite(acceleration).all(axis=1)]
        if len(complete) < 2:
            fail("Insufficient complete acceleration samples to check expected movement.")
            scores.append(0.0)
        else:
            # Remove a constant gravity/offset vector without rectifying the signal.
            dynamic = complete - np.mean(complete, axis=0)
            rms = float(np.sqrt(np.mean(np.sum(dynamic ** 2, axis=1))))
            measures["movement_rms_m_s2"] = rms
            scores.append(min(1.0, rms / q["movement_rms_min_m_s2"]))
            if rms < q["movement_rms_min_m_s2"]:
                fail(f"No sufficient movement for {task}: acceleration RMS {rms:.4f} m/s2.")
    return result()
