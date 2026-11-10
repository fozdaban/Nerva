"""Decode video incrementally and optionally run pretrained MediaPipe Tasks."""

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping

import numpy as np

from motor_pipeline.config import load_config
from .recording import LandmarkRecording


class VideoInputError(ValueError):
    """Video decoding, timestamps, model setup or identity is unusable."""


@dataclass
class FrameDetection:
    points: np.ndarray | None
    confidence: np.ndarray | None = None


class _MediaPipeDetector:
    """One participant only; ambiguous frames are missing rather than swapped."""

    def __init__(self, kind: str, settings: Mapping[str, Any]):
        model_path = settings[f"{kind}_model_path"]
        if not model_path or not Path(model_path).is_file():
            raise VideoInputError(f"Configure an existing local {kind}_model_path; models are never auto-downloaded")
        try:
            import mediapipe as mp
        except ImportError as exc:
            raise VideoInputError("MediaPipe is unavailable; install the optional video dependencies") from exc
        self.mp, self.kind = mp, kind
        options_type = mp.tasks.vision.HandLandmarkerOptions if kind == "hand" else mp.tasks.vision.PoseLandmarkerOptions
        landmarker_type = mp.tasks.vision.HandLandmarker if kind == "hand" else mp.tasks.vision.PoseLandmarker
        # Ask for two objects to detect ambiguity rather than silently switching
        # to whichever hand/person has the largest confidence on each frame.
        count = {"num_hands": 2} if kind == "hand" else {"num_poses": 2}
        for name in ("minimum_detection_confidence", "minimum_presence_confidence", "minimum_tracking_confidence"):
            if not np.isfinite(settings[name]) or not 0 <= settings[name] <= 1:
                raise ValueError(f"{name} must be in [0,1]")
        prefix = "hand" if kind == "hand" else "pose"
        thresholds = {f"min_{prefix}_detection_confidence": settings["minimum_detection_confidence"],
                      f"min_{prefix}_presence_confidence": settings["minimum_presence_confidence"],
                      "min_tracking_confidence": settings["minimum_tracking_confidence"]}
        options = options_type(base_options=mp.tasks.BaseOptions(model_asset_path=str(model_path)),
                               running_mode=mp.tasks.vision.RunningMode.VIDEO, **count, **thresholds)
        self.detector = landmarker_type.create_from_options(options)

    def __call__(self, rgb: np.ndarray, timestamp_ms: int) -> FrameDetection:
        image = self.mp.Image(image_format=self.mp.ImageFormat.SRGB, data=np.ascontiguousarray(rgb))
        result = self.detector.detect_for_video(image, timestamp_ms)
        sets = result.hand_landmarks if self.kind == "hand" else result.pose_landmarks
        if len(sets) != 1:
            return FrameDetection(None)
        points = np.array([[p.x, p.y, p.z] for p in sets[0]], dtype=float)
        if self.kind == "hand":
            return FrameDetection(points)  # Handedness scores are not detection confidence.
        confidence = []
        for p in sets[0]:
            available = [float(value) for value in (getattr(p, "visibility", None), getattr(p, "presence", None)) if value is not None]
            confidence.append(min(available) if available else np.nan)
        return FrameDetection(points, np.asarray(confidence))

    def close(self) -> None:
        self.detector.close()


def extract_landmarks(path: str | Path, kind: str, config: Mapping[str, Any] | None = None,
                      metadata: Mapping[str, Any] | None = None,
                      detector: Callable[[np.ndarray, int], FrameDetection] | None = None) -> LandmarkRecording:
    """Read every frame; no silent subsampling, caching or model downloads.

    A caller-owned detector callback allows deterministic testing without model
    inference. Rotation is explicit; callers must confirm phone orientation.
    Timestamp fallback to frame index/fps is opt-in, because it assumes CFR.
    """
    config = load_config() if config is None else config
    if kind not in ("hand", "pose"):
        raise ValueError("kind must be hand or pose")
    settings = config["video"]
    if not isinstance(settings["max_frames"], int) or settings["max_frames"] < 2:
        raise ValueError("max_frames must be an integer >=2")
    if not isinstance(settings["max_width_px"], int) or settings["max_width_px"] < 16:
        raise ValueError("max_width_px must be an integer >=16")
    if settings["rotation_degrees"] not in (0, 90, 180, 270):
        raise ValueError("rotation_degrees must be 0,90,180,270")
    try:
        import cv2
    except ImportError as exc:
        raise VideoInputError("OpenCV is unavailable; install the optional video dependencies") from exc
    capture = cv2.VideoCapture(str(path))
    owned = None
    try:
        if not capture.isOpened():
            raise VideoInputError(f"Cannot open video: {path}")
        capture_fps = float(capture.get(cv2.CAP_PROP_FPS))
        advertised_count = float(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        if settings["allow_constant_fps_timestamps"] and (not np.isfinite(capture_fps) or capture_fps <= 0):
            raise VideoInputError("Constant-fps timing requires a valid capture frame rate")
        if detector is None:
            owned = _MediaPipeDetector(kind, settings)
            detector = owned
        points, confidence, times = [], [], []
        count = 21 if kind == "hand" else 33
        width = height = None
        last_ms = -1
        timestamp_source = "frame_index_assuming_constant_fps" if settings["allow_constant_fps_timestamps"] else "decoder_presentation_time"
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            index = len(times)
            if index >= settings["max_frames"]:
                raise VideoInputError("Video exceeds max_frames; analysis was not silently truncated")
            if settings["rotation_degrees"]:
                frame = np.rot90(frame, k=-(settings["rotation_degrees"] // 90))
            current_height, current_width = frame.shape[:2]
            if width is None:
                width, height = current_width, current_height
            elif (current_width, current_height) != (width, height):
                raise VideoInputError("Frame dimensions changed during recording")
            time_s = index / capture_fps if settings["allow_constant_fps_timestamps"] else float(capture.get(cv2.CAP_PROP_POS_MSEC)) / 1000
            if not np.isfinite(time_s) or (times and time_s <= times[-1]):
                raise VideoInputError("Decoder timestamps are missing/nonmonotonic; CFR fallback requires explicit configuration")
            # MediaPipe requires strictly increasing integer milliseconds.
            timestamp_ms = int(round(time_s * 1000))
            if timestamp_ms <= last_ms:
                raise VideoInputError("Frame timestamps cannot be represented as increasing integer milliseconds")
            last_ms = timestamp_ms
            if current_width > settings["max_width_px"]:
                scale = settings["max_width_px"] / current_width
                frame = cv2.resize(frame, (settings["max_width_px"], max(1, int(round(current_height * scale)))))
            rgb = cv2.cvtColor(np.ascontiguousarray(frame), cv2.COLOR_BGR2RGB)
            processed_height, processed_width = rgb.shape[:2]
            detection = detector(rgb, timestamp_ms)
            array = np.full((count, 3), np.nan) if detection.points is None else np.asarray(detection.points, dtype=float)
            if array.shape != (count, 3):
                raise VideoInputError("Detector returned the wrong landmark shape")
            points.append(array)
            if detection.confidence is None:
                confidence.append(np.full(count, np.nan))
            else:
                value = np.asarray(detection.confidence, dtype=float)
                if value.shape != (count,):
                    raise VideoInputError("Detector returned the wrong confidence shape")
                confidence.append(value)
            times.append(time_s)
        if len(times) < 2:
            raise VideoInputError("Video contains fewer than two decodable frames")
        if np.isfinite(advertised_count) and advertised_count >= 2 and len(times) + 1 < advertised_count:
            raise VideoInputError("Video decoding ended before its advertised frame count; incomplete counts are withheld")
        values = np.stack(confidence)
        has_confidence = np.isfinite(values).any()
        time_values = np.asarray(times) - times[0]
        return LandmarkRecording(time_values, np.stack(points), kind, processed_width, processed_height,
                                  values if has_confidence else None,
                                  {**(metadata or {}), "capture_fps": capture_fps if np.isfinite(capture_fps) else None,
                                   "timestamp_source": timestamp_source, "rotation_degrees": settings["rotation_degrees"],
                                   "decoded_width_after_rotation": width, "decoded_height_after_rotation": height},
                                  {"decoded_frames": len(times), "subsampled": False,
                                   "advertised_frame_count": int(advertised_count) if np.isfinite(advertised_count) and advertised_count > 0 else None})
    finally:
        capture.release()
        if owned is not None:
            owned.close()
