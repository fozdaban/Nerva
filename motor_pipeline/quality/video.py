"""Conservative video quality gates using observed landmark/frame evidence."""

from dataclasses import dataclass
from typing import Any, Mapping

import numpy as np

from motor_pipeline.config import load_config
from motor_pipeline.vision.recording import LandmarkRecording


@dataclass
class VideoQuality:
    score: float
    reasons: list[str]
    valid_frames: np.ndarray
    measurements: dict[str, Any]

    @property
    def status(self) -> str:
        return "retest" if self.reasons else "ok"

    def as_dict(self) -> dict[str, Any]:
        return {"score": self.score, "reasons": list(self.reasons)}

    def retest_response(self) -> dict[str, Any] | None:
        return {"status": "retest", "reasons": list(self.reasons)} if self.reasons else None


def assess_video(recording: LandmarkRecording, task: str, config: Mapping[str, Any] | None = None,
                 required_indices: list[int] | None = None) -> VideoQuality:
    """Missing/cropped/low-visibility landmarks cannot silently supply metrics.

    Hand landmark detection confidence is not exposed by the Tasks result.
    Its mean remains None; handedness is never substituted for it.
    """
    config = load_config() if config is None else config
    q, v = config["quality"], config["video"]
    if task not in ("tapping_video", "gait_video"):
        raise ValueError("Unknown video task")
    expected_kind = "hand" if task == "tapping_video" else "pose"
    if recording.kind != expected_kind:
        raise ValueError(f"{task} requires {expected_kind} landmarks")
    indices = required_indices if required_indices is not None else (
        [0, 4, 8, 9] if recording.kind == "hand" else [11, 12, 15, 16, 23, 24, 27, 28])
    if not indices or len(set(indices)) != len(indices) or min(indices) < 0 or max(indices) >= recording.points.shape[1]:
        raise ValueError("required_indices must be unique valid landmark indices")
    for key in ("minimum_score", "minimum_landmark_fraction", "minimum_landmark_confidence"):
        if not np.isfinite(q[key]) or not 0 < q[key] <= 1:
            raise ValueError(f"{key} must be in (0, 1]")
    if not 0 <= v["minimum_inside_margin"] < .5 or not 0 <= v["maximum_out_of_frame_fraction"] <= 1:
        raise ValueError("Invalid video frame boundary thresholds")
    if not np.isfinite(v["maximum_frame_gap_s"]) or v["maximum_frame_gap_s"] <= 0:
        raise ValueError("maximum_frame_gap_s must be positive")
    if not np.isfinite(q["minimum_video_fps"]) or q["minimum_video_fps"] <= 0:
        raise ValueError("minimum_video_fps must be positive")
    minimum_duration = q["minimum_duration_s"][task]
    if not np.isfinite(minimum_duration) or minimum_duration <= 0:
        raise ValueError("Minimum video duration must be positive")
    reasons, measures = [], {"mean_landmark_confidence": None}
    n = len(recording.t)
    valid = np.zeros(n, dtype=bool)

    def finish(score: float) -> VideoQuality:
        score = float(np.clip(score, 0, 1))
        if reasons:
            score = min(score, q["minimum_score"] / 2)
        if score < q["minimum_score"] and not reasons:
            reasons.append("Video quality is below the configured threshold.")
        return VideoQuality(score, reasons, valid, measures)

    if n < 2 or not np.isfinite(recording.t).all() or np.any(np.diff(recording.t) <= 0):
        reasons.append("At least two finite, strictly increasing frame timestamps are required.")
        return finish(0)
    dt = np.diff(recording.t)
    fps = float(1 / np.median(dt))
    duration = float(recording.t[-1] - recording.t[0] + 1 / fps)
    measures.update(processed_fps=fps, duration_s=duration,
                    capture_fps=recording.metadata.get("capture_fps"),
                    timestamp_source=recording.metadata.get("timestamp_source", "provided"))
    if fps + 1e-9 < q["minimum_video_fps"]:
        reasons.append(f"Frame rate is too low: {fps:.2f} fps.")
    if duration + 1e-9 < minimum_duration:
        reasons.append(f"Video is too short for {task}: {duration:.3f} s; need {minimum_duration:g} s.")
    longest_gap = float(dt.max())
    measures["longest_frame_gap_s"] = longest_gap
    if longest_gap > v["maximum_frame_gap_s"] + 1e-9:
        reasons.append(f"Video timestamp gap is too long: {longest_gap:.3f} s.")
    points = recording.points[:, indices, :2]
    tracked = np.isfinite(points).all(axis=(1, 2))
    margin = v["minimum_inside_margin"]
    inside = ((points >= margin) & (points <= 1 - margin)).all(axis=(1, 2))
    cropped = tracked & ~inside
    tracked_fraction = float(tracked.mean())
    cropped_fraction = float(cropped.mean())
    valid[:] = tracked & inside
    if recording.confidence is not None:
        confidence = recording.confidence[:, indices]
        finite_confidence = np.isfinite(confidence).all(axis=1)
        confidence_in_range = ((confidence >= 0) & (confidence <= 1)).all(axis=1)
        available = tracked & finite_confidence & confidence_in_range
        mean_confidence = float(confidence[available].mean()) if available.any() else None
        measures["mean_landmark_confidence"] = mean_confidence
        valid &= available & (confidence >= q["minimum_landmark_confidence"]).all(axis=1)
        if mean_confidence is None or mean_confidence < q["minimum_landmark_confidence"]:
            reasons.append("Reported landmark visibility/presence is too low or unavailable.")
    elif recording.kind == "pose":
        # Pose features require the per-landmark visibility evidence normally
        # supplied by the Pose Tasks result. Never assume 1.0 confidence.
        valid[:] = False
        reasons.append("Pose landmark visibility is unavailable.")
    valid_fraction = float(valid.mean())
    measures.update(tracked_fraction=tracked_fraction, out_of_frame_fraction=cropped_fraction,
                    valid_frame_fraction=valid_fraction,
                    confidence_source="not_exposed" if recording.confidence is None else "landmark_visibility_or_presence")
    if tracked_fraction < q["minimum_landmark_fraction"]:
        reasons.append(f"Landmarks tracked in only {tracked_fraction:.1%} of frames.")
    if valid_fraction < q["minimum_landmark_fraction"]:
        reasons.append(f"Only {valid_fraction:.1%} of frames have complete usable landmarks.")
    if cropped_fraction > v["maximum_out_of_frame_fraction"]:
        reasons.append(f"Required landmarks are partly out of frame in {cropped_fraction:.1%} of frames.")
    return finish(min(valid_fraction, min(1, fps / q["minimum_video_fps"]), min(1, duration / minimum_duration)))
