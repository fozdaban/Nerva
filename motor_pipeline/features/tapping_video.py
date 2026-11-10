"""Thumb/index image-plane tapping cycles normalized by observed hand size."""

from typing import Any, Mapping

import numpy as np
from scipy.signal import find_peaks

from motor_pipeline.config import load_config
from motor_pipeline.quality.video import assess_video
from motor_pipeline.vision.recording import LandmarkRecording
from motor_pipeline.vision.windows import feature_spread, landmark_windows, minimum_windows


def _tap_metrics(times: np.ndarray, amplitudes: np.ndarray, intervals: np.ndarray,
                 multiplier: float, trend: bool = True) -> dict[str, Any]:
    mean = float(intervals.mean()) if len(intervals) else None
    slope = decrement = None
    if trend and len(times) >= 3 and times[-1] > times[0]:
        fitted_slope, intercept = np.polyfit(times - times[0], amplitudes, 1)
        slope = float(fitted_slope)
        decrement = float(-slope * (times[-1] - times[0]) / intercept) if intercept > 0 else None
    return {"tap_count": int(len(times)), "tap_rate_hz": 1 / mean if mean else None,
            "inter_tap_interval_cv": float(np.std(intervals, ddof=1) / mean) if len(intervals) >= 2 and mean else None,
            "median_tap_amplitude_hand_lengths": float(np.median(amplitudes)),
            "amplitude_slope_hand_lengths_per_s": slope, "amplitude_decrement_fraction": decrement,
            "hesitation_count": int(np.sum(intervals > multiplier * np.median(intervals))) if len(intervals) else None}


def extract_tapping_video(recording: LandmarkRecording, config: Mapping[str, Any] | None = None) -> dict[str, Any]:
    """Analyze extracted hand landmarks; no optional video libraries are needed."""
    config = load_config() if config is None else config
    s = config["features"]["tapping_video"]
    for key in ("minimum_peak_distance_s", "minimum_peak_prominence", "minimum_hand_size"):
        if not np.isfinite(s[key]) or s[key] <= 0:
            raise ValueError(f"{key} must be positive")
    if not np.isfinite(s["hesitation_median_multiplier"]) or s["hesitation_median_multiplier"] <= 1:
        raise ValueError("hesitation_median_multiplier must exceed 1")
    if not 0 < s["minimum_valid_window_fraction"] <= 1:
        raise ValueError("minimum_valid_window_fraction must be in (0,1]")
    if not isinstance(s["minimum_peaks_per_window"], int) or s["minimum_peaks_per_window"] < 3:
        raise ValueError("minimum_peaks_per_window must be an integer >=3")
    hand = s["hand_size_landmarks"]
    tips = s["distance_landmarks"]
    if len(hand) != 2 or len(tips) != 2 or min(*hand, *tips) < 0 or max(*hand, *tips) >= 21:
        raise ValueError("Hand-size and tip pairs need two valid landmark indices")
    required = sorted(set([*hand, *tips]))
    q = assess_video(recording, "tapping_video", config, required)
    if q.status == "retest":
        return q.retest_response()
    xy = recording.image_xy()
    size = np.linalg.norm(xy[:, hand[0]] - xy[:, hand[1]], axis=1)
    distance = np.linalg.norm(xy[:, tips[0]] - xy[:, tips[1]], axis=1)
    valid = q.valid_frames & np.isfinite(size) & (size >= s["minimum_hand_size"])
    signal = np.divide(distance, size, out=np.full_like(distance, np.nan), where=size > 0)
    windows, expected, fs = landmark_windows(recording.t, valid, config)
    rows, detections = [], {}
    for segment_id, indices in windows:
        peaks, properties = find_peaks(signal[indices], prominence=s["minimum_peak_prominence"],
                                       distance=max(1, int(np.ceil(s["minimum_peak_distance_s"] * fs))))
        absolute = indices[peaks]
        # A recording/window can begin mid-cycle. Use the larger observed
        # peak-to-side-base range for amplitude instead of forcing that first
        # event's half-prominence into a full-cycle amplitude trend.
        local = signal[indices]
        amplitudes = np.maximum(local[peaks] - local[properties["left_bases"]],
                                local[peaks] - local[properties["right_bases"]])
        refined_times = []
        for index, peak, amplitude in zip(absolute, peaks, amplitudes):
            a, b, c = local[peak - 1:peak + 2]
            denominator = a - 2 * b + c
            offset = float(np.clip(.5 * (a - c) / denominator, -.5, .5)) if denominator < 0 else 0.0
            time = float(recording.t[index] + offset / fs)
            refined_times.append(time)
            key = (segment_id, int(index))
            detections.setdefault(key, []).append((time, float(amplitude)))
        if len(peaks) < s["minimum_peaks_per_window"]:
            continue
        times = np.asarray(refined_times)
        rows.append({"start_s": float(recording.t[indices[0]]), "segment_id": segment_id,
                     "features": _tap_metrics(times, amplitudes, np.diff(times), s["hesitation_median_multiplier"]),
                     "event_times_s": times.tolist()})
    if len(rows) < minimum_windows(config) or not expected or len(rows) / expected < s["minimum_valid_window_fraction"]:
        return {"status": "retest", "reasons": [f"Too few complete video tapping windows: {len(rows)} valid of {expected} expected; check tracking, hand scale or movement."]}
    ordered = sorted(detections)
    segments = np.array([key[0] for key in ordered])
    times = np.array([np.median([event[0] for event in detections[key]]) for key in ordered])
    amplitudes = np.array([np.median([event[1] for event in detections[key]]) for key in ordered])
    intervals = np.diff(times)[np.diff(segments) == 0]
    if len(intervals) < 2:
        return {"status": "retest", "reasons": ["Too few within-segment video tap intervals."]}
    features = _tap_metrics(times, amplitudes, intervals, s["hesitation_median_multiplier"], len(set(segments)) == 1)
    features["tracked_frame_fraction"] = q.measurements["tracked_fraction"]
    for row in rows:
        row["features"]["tracked_frame_fraction"] = 1.0
    return {"status": "ok", "quality": q.as_dict(), "features": features,
            "per_window": rows, "spread": feature_spread(rows, list(features), q.score),
            "n_valid_windows": len(rows), "expected_windows": expected,
            "video_quality": q.measurements, "event_times_s": times.tolist(),
            "event_amplitudes": amplitudes.tolist(), "event_segment_ids": segments.tolist(),
            "inter_tap_intervals_s": intervals.tolist(), "limitations": [
                "A maximal thumb/index opening is a tapping-cycle proxy, not proof of finger contact.",
                "Distances use aspect-corrected 2D image coordinates normalized by wrist-to-MCP size; perspective and out-of-plane motion remain confounders.",
                "Hand detection confidence is unavailable in MediaPipe Tasks output; handedness scores are not substituted.",
                "The same intended hand must remain in view; persistent identity through occlusion or single-hand switches is unvalidated.",
                "Missing frames separate segments; cross-segment pauses and global amplitude trends are withheld.",
                "Window IQRs are descriptive spread, not calibrated confidence intervals."
            ]}
