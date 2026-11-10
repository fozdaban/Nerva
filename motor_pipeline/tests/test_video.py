"""Deterministic image-plane landmarks; no claim of real camera accuracy."""

import json
import sys
from types import SimpleNamespace

import numpy as np
import pytest

from motor_pipeline.config import load_config
from motor_pipeline.features.tapping_video import extract_tapping_video
from motor_pipeline.features.gait_video import extract_gait_video
from motor_pipeline.quality.video import assess_video
from motor_pipeline.vision import FrameDetection, LandmarkRecording, VideoInputError, extract_landmarks
from motor_pipeline.vision.extract import _MediaPipeDetector


def hand_fixture(fps=30, duration=20, rate=2, decrement=0, width=1280, height=720):
    t = np.arange(int(fps * duration)) / fps
    points = np.full((len(t), 21, 3), .5)
    points[:, :, 2] = 0
    points[:, 0, :2] = [.4, .5]
    points[:, 9, :2] = [.4, .65]
    points[:, 4, :2] = [.4, .4]
    distance = .1 + .075 * (1 - decrement * t / duration) * (1 + np.sin(2 * np.pi * rate * t))
    points[:, 8, 0] = .4 + distance / (width / height)
    points[:, 8, 1] = .4
    return LandmarkRecording(t, points, "hand", width, height)


def pose_fixture(fps=30, duration=30, cadence=120):
    t = np.arange(int(fps * duration)) / fps
    phase = 2 * np.pi * cadence / 120 * t
    points = np.full((len(t), 33, 3), .5)
    points[:, :, 2] = 0
    for index, xy in {11: [.4, .3], 12: [.6, .3], 23: [.4, .6], 24: [.6, .6]}.items():
        points[:, index, :2] = xy
    points[:, 27, :2] = [.45, .8]
    points[:, 28, :2] = [.55, .8]
    points[:, 27, 1] += .05 * np.sin(phase)
    points[:, 28, 1] -= .05 * np.sin(phase)
    points[:, 15, :2] = [.3, .5]
    points[:, 16, :2] = [.7, .5]
    points[:, 15, 0] += .03 * np.sin(phase)
    points[:, 16, 0] -= .03 * np.sin(phase)
    return LandmarkRecording(t, points, "pose", 1000, 1000,
                              np.full((len(t), 33), .95))


@pytest.mark.parametrize("fps", [24, 30, 60])
@pytest.mark.parametrize("rate", [1.5, 2, 3])
def test_tapping_known_rate_and_count(fps, rate):
    result = extract_tapping_video(hand_fixture(fps=fps, rate=rate))
    assert result["status"] == "ok"
    assert result["features"]["tap_rate_hz"] == pytest.approx(rate, rel=.015)
    assert result["features"]["tap_count"] == round(20 * rate)
    assert result["features"]["inter_tap_interval_cv"] < .05
    assert result["features"]["hesitation_count"] == 0
    assert result["video_quality"]["mean_landmark_confidence"] is None


def test_aspect_ratio_correction_matches_same_physical_geometry():
    first = extract_tapping_video(hand_fixture(width=720, height=720))
    second = extract_tapping_video(hand_fixture(width=1280, height=720))
    assert first["features"]["median_tap_amplitude_hand_lengths"] == pytest.approx(second["features"]["median_tap_amplitude_hand_lengths"], rel=1e-6)


def test_video_amplitude_decrement():
    result = extract_tapping_video(hand_fixture(decrement=.4))
    assert result["features"]["amplitude_slope_hand_lengths_per_s"] < 0
    assert result["features"]["amplitude_decrement_fraction"] == pytest.approx(.4, abs=.04)


def test_tapping_counts_are_not_sum_of_overlap():
    result = extract_tapping_video(hand_fixture())
    assert result["features"]["tap_count"] == 40
    assert sum(row["features"]["tap_count"] for row in result["per_window"]) > 40


@pytest.mark.parametrize("cadence", [80, 120, 160])
def test_video_gait_known_ankle_trajectories(cadence):
    result = extract_gait_video(pose_fixture(cadence=cadence))
    assert result["status"] == "ok"
    assert result["features"]["cadence_steps_min"] == pytest.approx(cadence, abs=2)
    assert result["features"]["mean_step_time_s"] == pytest.approx(60 / cadence, abs=.02)
    assert result["features"]["step_time_cv"] < .04
    assert result["features"]["step_count"] == round(cadence / 2)
    assert result["features"]["arm_swing_amplitude_body_lengths"] == pytest.approx(.2, abs=.02)


def test_synchronous_ankle_peaks_are_ambiguous_not_double_counted():
    recording = pose_fixture()
    recording.points[:, 28, 1] = recording.points[:, 27, 1]
    assert extract_gait_video(recording)["status"] == "retest"


def test_unilateral_ankle_activity_retests_instead_of_reporting_bilateral_cadence():
    recording = pose_fixture()
    recording.points[:, 28, 1] = .8
    assert extract_gait_video(recording)["status"] == "retest"


def test_too_small_body_scale_retests_instead_of_amplifying_noise():
    recording = pose_fixture()
    recording.points[:, 11, :2] = recording.points[:, 23, :2] + .0001
    assert extract_gait_video(recording)["status"] == "retest"


def test_gait_intervals_do_not_bridge_unsupported_motion_holes():
    recording = pose_fixture(duration=40)
    # Visible, finite landmarks but no ankle activity for a prolonged interval.
    # This is not a tracking gap: accepted-motion windows must bound timing.
    quiet = (recording.t >= 15) & (recording.t < 23)
    recording.points[quiet, 27, 1] = .8
    recording.points[quiet, 28, 1] = .8
    result = extract_gait_video(recording)
    assert result["status"] == "ok"
    assert max(result["step_intervals_s"]) < 1


def test_quality_tracks_only_relevant_landmarks_and_never_invents_hand_confidence():
    recording = hand_fixture()
    recording.points[:, 20, 0] = 2  # Not used by thumb/index metric.
    q = assess_video(recording, "tapping_video")
    assert q.status == "ok"
    assert q.measurements["mean_landmark_confidence"] is None
    assert q.measurements["tracked_fraction"] == 1


@pytest.mark.parametrize("problem", ["missing", "cropped", "slow", "short", "bad_time", "large_gap"])
def test_video_quality_problems_retest_without_metrics(problem):
    recording = hand_fixture()
    if problem == "missing":
        recording.points[:200] = np.nan
    elif problem == "cropped":
        recording.points[:100, 8, 0] = 1.1
    elif problem == "slow":
        recording = hand_fixture(fps=15)
    elif problem == "short":
        recording = hand_fixture(duration=5)
    elif problem == "bad_time":
        recording.t[1] = recording.t[0]
    else:
        recording.t[300:] += 1
    result = extract_tapping_video(recording)
    assert result["status"] == "retest"
    assert set(result) == {"status", "reasons"}


def test_pose_visibility_not_handedness_confidence_is_gated():
    recording = pose_fixture()
    recording.confidence[:] = .1
    assert extract_gait_video(recording)["status"] == "retest"
    recording.confidence = None
    assert extract_gait_video(recording)["status"] == "retest"


def test_missing_frames_are_not_hesitations_or_cross_segment_trends():
    recording = hand_fixture(duration=30)
    recording.points[450:453] = np.nan
    result = extract_tapping_video(recording)
    assert result["status"] == "ok"
    assert result["features"]["hesitation_count"] == 0
    assert result["features"]["amplitude_decrement_fraction"] is None
    assert len(set(result["event_segment_ids"])) == 2


def test_frequent_missing_frames_retest_despite_high_tracking_fraction():
    recording = hand_fixture()
    recording.points[np.arange(60, len(recording.t), 60)] = np.nan
    result = extract_tapping_video(recording)
    assert result["status"] == "retest"


def test_stationary_hand_and_too_small_hand_retest():
    recording = hand_fixture()
    recording.points[:, 8] = recording.points[0, 8]
    assert extract_tapping_video(recording)["status"] == "retest"
    recording = hand_fixture()
    recording.points[:, 9] = recording.points[:, 0]
    assert extract_tapping_video(recording)["status"] == "retest"


def test_video_outputs_are_json_safe_with_auditable_spread():
    for result in (extract_tapping_video(hand_fixture()), extract_gait_video(pose_fixture())):
        json.dumps(result, allow_nan=False)
        assert set(result["features"]) == set(result["spread"])
        assert result["spread"]["tracked_frame_fraction"]["quality_score"] == result["quality"]["score"]


class FakeCapture:
    def __init__(self, cv2, n=4, repeated_times=False, advertised=4):
        self.cv2, self.n, self.index = cv2, n, 0
        self.repeated_times, self.released, self.advertised = repeated_times, False, advertised

    def isOpened(self):
        return True

    def get(self, key):
        return {self.cv2.CAP_PROP_FPS: 30, self.cv2.CAP_PROP_FRAME_COUNT: self.advertised,
                self.cv2.CAP_PROP_POS_MSEC: 0 if self.repeated_times else (self.index - 1) * 1000 / 30}.get(key, 0)

    def read(self):
        self.index += 1
        return (True, np.zeros((120, 160, 3), dtype=np.uint8)) if self.index <= self.n else (False, None)

    def release(self):
        self.released = True


def test_injected_extraction_preserves_missing_frames_and_releases_capture(monkeypatch):
    cv2 = pytest.importorskip("cv2")
    capture = FakeCapture(cv2)
    monkeypatch.setattr(cv2, "VideoCapture", lambda path: capture)
    calls = []

    def detector(rgb, milliseconds):
        calls.append(milliseconds)
        return FrameDetection(None if len(calls) == 2 else np.full((21, 3), .5))

    result = extract_landmarks("fake.mp4", "hand", detector=detector)
    assert calls == [0, 33, 67, 100]
    assert len(result.t) == 4
    assert np.isnan(result.points[1]).all()
    assert result.confidence is None
    assert result.metadata["capture_fps"] == 30
    assert result.diagnostics["decoded_frames"] == 4
    assert capture.released


@pytest.mark.parametrize("problem", ["timestamps", "max_frames", "incomplete"])
def test_extraction_refuses_silent_timing_or_truncation(monkeypatch, problem):
    cv2 = pytest.importorskip("cv2")
    capture = FakeCapture(cv2, repeated_times=problem == "timestamps", advertised=100 if problem == "incomplete" else 4)
    monkeypatch.setattr(cv2, "VideoCapture", lambda path: capture)
    cfg = load_config()
    if problem == "max_frames":
        cfg["video"]["max_frames"] = 2
    with pytest.raises(VideoInputError):
        extract_landmarks("fake.mp4", "hand", cfg, detector=lambda rgb, t: FrameDetection(None))
    assert capture.released


def test_cfr_timing_is_explicit_and_recorded(monkeypatch):
    cv2 = pytest.importorskip("cv2")
    capture = FakeCapture(cv2, repeated_times=True)
    monkeypatch.setattr(cv2, "VideoCapture", lambda path: capture)
    cfg = load_config()
    cfg["video"]["allow_constant_fps_timestamps"] = True
    result = extract_landmarks("fake.mp4", "hand", cfg, detector=lambda rgb, t: FrameDetection(None))
    assert result.metadata["timestamp_source"] == "frame_index_assuming_constant_fps"
    assert result.t[1] == pytest.approx(1 / 30)


def test_hand_detector_never_uses_handedness_score_as_confidence():
    wrapper = _MediaPipeDetector.__new__(_MediaPipeDetector)
    wrapper.kind = "hand"
    wrapper.mp = SimpleNamespace(Image=lambda **kwargs: kwargs, ImageFormat=SimpleNamespace(SRGB="rgb"))
    landmark = SimpleNamespace(x=.5, y=.5, z=0)
    result = SimpleNamespace(hand_landmarks=[[landmark] * 21], handedness=[[SimpleNamespace(score=.99)]])
    wrapper.detector = SimpleNamespace(detect_for_video=lambda image, time: result)
    detected = wrapper(np.zeros((4, 4, 3), dtype=np.uint8), 0)
    assert detected.confidence is None
    result.hand_landmarks *= 2
    assert wrapper(np.zeros((4, 4, 3), dtype=np.uint8), 1).points is None


def test_missing_local_model_fails_without_download():
    cfg = load_config()
    cfg["video"]["hand_model_path"] = None
    with pytest.raises(VideoInputError, match="never auto-downloaded"):
        _MediaPipeDetector("hand", cfg["video"])


@pytest.mark.parametrize("kind", ["hand", "pose"])
def test_tasks_constructor_uses_video_mode_and_configured_thresholds(monkeypatch, tmp_path, kind):
    asset = tmp_path / "model.task"
    asset.write_bytes(b"mock only; not model weights")
    cfg = load_config()
    cfg["video"][f"{kind}_model_path"] = str(asset)
    calls = []
    factory = SimpleNamespace(create_from_options=lambda options: (calls.append(options), SimpleNamespace(close=lambda: None))[1])
    vision = SimpleNamespace(HandLandmarkerOptions=lambda **kwargs: kwargs,
                             PoseLandmarkerOptions=lambda **kwargs: kwargs,
                             HandLandmarker=factory, PoseLandmarker=factory,
                             RunningMode=SimpleNamespace(VIDEO="video"))
    fake = SimpleNamespace(tasks=SimpleNamespace(BaseOptions=lambda **kwargs: kwargs, vision=vision))
    monkeypatch.setitem(sys.modules, "mediapipe", fake)
    detector = _MediaPipeDetector(kind, cfg["video"])
    detector.close()
    assert calls[0]["running_mode"] == "video"
    assert calls[0][f"min_{kind}_presence_confidence"] == .6
    assert calls[0]["min_tracking_confidence"] == .6
    assert calls[0]["base_options"]["model_asset_path"] == str(asset)


def test_actual_opencv_decode_with_injected_detector(tmp_path):
    cv2 = pytest.importorskip("cv2")
    path = tmp_path / "synthetic.avi"
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"MJPG"), 30, (160, 120))
    if not writer.isOpened():
        pytest.skip("Local OpenCV build does not support the test AVI codec")
    try:
        for _ in range(60):
            writer.write(np.zeros((120, 160, 3), dtype=np.uint8))
    finally:
        writer.release()
    cfg = load_config()
    cfg["video"].update(rotation_degrees=90, max_width_px=64)
    shape = []
    recording = extract_landmarks(path, "hand", cfg,
                                 detector=lambda rgb, t: (shape.append(rgb.shape), FrameDetection(None))[1])
    assert len(recording.t) == 60
    assert recording.t[-1] == pytest.approx(59 / 30, abs=.001)
    assert recording.width == 64
    assert recording.height == 85
    assert shape[0] == (85, 64, 3)
    assert recording.diagnostics["subsampled"] is False
