"""Image-plane ankle-event timing and body-normalized wrist swing proxies."""

from typing import Any, Mapping

import numpy as np
from scipy.signal import find_peaks

from motor_pipeline.config import load_config
from motor_pipeline.quality.video import assess_video
from motor_pipeline.vision.recording import LandmarkRecording
from motor_pipeline.vision.windows import feature_spread, landmark_windows, minimum_windows


def _metrics(times: np.ndarray, intervals: np.ndarray, swing: float) -> dict[str, Any]:
    mean = float(intervals.mean()) if len(intervals) else None
    return {"step_count": int(len(times)), "cadence_steps_min": 60 / mean if mean else None,
            "mean_step_time_s": mean,
            "step_time_cv": float(np.std(intervals, ddof=1) / mean) if len(intervals) >= 2 and mean else None,
            "arm_swing_amplitude_body_lengths": swing}


def extract_gait_video(recording: LandmarkRecording, config: Mapping[str, Any] | None = None) -> dict[str, Any]:
    """Interpret ankle trajectory peaks as step proxies; no metric distance/speed."""
    config = load_config() if config is None else config
    s = config["features"]["gait_video"]
    if s["ankle_axis"] not in ("x", "y"):
        raise ValueError("ankle_axis must be x or y")
    for key in ("minimum_peak_distance_s", "minimum_peak_prominence", "event_merge_tolerance_s", "minimum_body_size"):
        if not np.isfinite(s[key]) or s[key] <= 0:
            raise ValueError(f"{key} must be positive")
    if s["event_merge_tolerance_s"] >= s["minimum_peak_distance_s"] / 2:
        raise ValueError("event_merge_tolerance_s is too large relative to peak distance")
    if not isinstance(s["minimum_steps_per_window"], int) or s["minimum_steps_per_window"] < 3:
        raise ValueError("minimum_steps_per_window must be an integer >=3")
    if not 0 < s["minimum_valid_window_fraction"] <= 1:
        raise ValueError("minimum_valid_window_fraction must be in (0,1]")
    scale_pair = s["body_scale_landmarks"]
    if len(scale_pair) != 2 or min(scale_pair) < 0 or max(scale_pair) >= 33:
        raise ValueError("body_scale_landmarks must contain two valid pose indices")
    required = sorted(set([11, 12, 15, 16, 23, 24, 27, 28, *scale_pair]))
    q = assess_video(recording, "gait_video", config, required)
    if q.status == "retest":
        return q.retest_response()
    xy = recording.image_xy()
    scale = np.linalg.norm(xy[:, scale_pair[0]] - xy[:, scale_pair[1]], axis=1)
    valid = q.valid_frames & np.isfinite(scale) & (scale >= s["minimum_body_size"])
    hip = (xy[:, 23] + xy[:, 24]) / 2
    coordinate = 0 if s["ankle_axis"] == "x" else 1
    signals = np.divide(xy[:, [27, 28], coordinate] - hip[:, coordinate, None], scale[:, None],
                         out=np.full((len(scale), 2), np.nan), where=scale[:, None] > 0)
    windows, expected, fs = landmark_windows(recording.t, valid, config)
    rows, detections = [], set()
    for segment_id, indices in windows:
        events = []
        for ankle in (0, 1):
            peaks, _ = find_peaks(signals[indices, ankle], prominence=s["minimum_peak_prominence"],
                                  distance=max(1, int(np.ceil(s["minimum_peak_distance_s"] * fs))))
            for index in indices[peaks]:
                detections.add((segment_id, int(index), ankle))
                events.append((float(recording.t[index]), ankle))
        events.sort()
        times = np.array([event[0] for event in events])
        if len(times) < s["minimum_steps_per_window"]:
            continue
        if np.any(np.diff(times) <= s["event_merge_tolerance_s"]):
            continue  # simultaneous ankle peaks are ambiguous in this projection
        if any(first[1] == last[1] for first, last in zip(events[:-1], events[1:])):
            continue  # unilateral or missed contralateral events are ambiguous
        swings = []
        for wrist in (15, 16):
            relative = (xy[indices, wrist] - hip[indices]) / scale[indices, None]
            swings.append(float(np.linalg.norm(np.percentile(relative, 95, axis=0) - np.percentile(relative, 5, axis=0))))
        rows.append({"start_s": float(recording.t[indices[0]]), "segment_id": segment_id,
                     "features": _metrics(times, np.diff(times), float(np.median(swings))),
                     "event_times_s": times.tolist()})
    if len(rows) < minimum_windows(config) or not expected or len(rows) / expected < s["minimum_valid_window_fraction"]:
        return {"status": "retest", "reasons": [f"Too few unambiguous video gait windows: {len(rows)} valid of {expected} expected."]}
    # Restrict global events to times actually covered by supported windows.
    supported = [(row["segment_id"], row["start_s"], row["start_s"] + config["preprocess"]["window_s"]) for row in rows]
    ordered = sorted(event for event in detections if any(segment == event[0] and start <= recording.t[event[1]] < end for segment, start, end in supported))
    times = np.array([recording.t[index] for _, index, _ in ordered])
    segments = np.array([segment for segment, _, _ in ordered])
    covered: list[tuple[int, float, float]] = []
    for segment, start, end in sorted(supported):
        if covered and covered[-1][0] == segment and start <= covered[-1][2] + 1e-9:
            previous = covered[-1]
            covered[-1] = (segment, previous[1], max(previous[2], end))
        else:
            covered.append((segment, start, end))
    intervals = []
    for i, (first, last) in enumerate(zip(times[:-1], times[1:])):
        if segments[i] != segments[i + 1] or not any(segment == segments[i] and start <= first and last < end + 1e-9 for segment, start, end in covered):
            continue
        if ordered[i][2] == ordered[i + 1][2]:
            return {"status": "retest", "reasons": ["Unilateral or missed alternating ankle events make step timing ambiguous."]}
        intervals.append(last - first)
    intervals = np.asarray(intervals)
    if len(intervals) < 2 or np.any(intervals <= s["event_merge_tolerance_s"]):
        return {"status": "retest", "reasons": ["Ambiguous simultaneous or insufficient video ankle events."]}
    swing = float(np.median([row["features"]["arm_swing_amplitude_body_lengths"] for row in rows]))
    features = _metrics(times, intervals, swing)
    features["tracked_frame_fraction"] = q.measurements["tracked_fraction"]
    for row in rows:
        row["features"]["tracked_frame_fraction"] = 1.0
    return {"status": "ok", "quality": q.as_dict(), "features": features,
            "per_window": rows, "spread": feature_spread(rows, list(features), q.score),
            "n_valid_windows": len(rows), "expected_windows": expected,
            "video_quality": q.measurements, "event_times_s": times.tolist(),
            "event_segment_ids": segments.tolist(), "step_intervals_s": intervals.tolist(),
            "limitations": [
                "Ankle image-trajectory peaks are step proxies, not validated heel-strike events.",
                "Camera view, occlusion, motion blur and perspective can change event counts; no physical speed or distance is inferred.",
                "The same participant must remain in view; persistent identity through occlusion or single-person switches is unvalidated.",
                "Arm swing is a 2D body-normalized image-plane range proxy, not anatomical joint angle.",
                "Missing frames separate segments; intervals across those gaps are excluded.",
                "Window IQRs describe spread and are not calibrated confidence intervals."
            ]}
