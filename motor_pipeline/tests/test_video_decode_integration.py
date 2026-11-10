"""Exercise real OpenCV I/O with an injected detector; not model validation."""

import numpy as np
import pytest

from motor_pipeline.config import load_config
from motor_pipeline.vision.extract import FrameDetection, extract_landmarks


def test_real_codec_roundtrip_keeps_all_frame_times(tmp_path):
    cv2 = pytest.importorskip("cv2", reason="Optional OpenCV is not installed")
    path = tmp_path / "generated.avi"
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"MJPG"), 30, (96, 64))
    if not writer.isOpened():
        pytest.skip("MJPG writer is unavailable on this platform")
    try:
        for frame_number in range(60):
            writer.write(np.full((64, 96, 3), frame_number, dtype=np.uint8))
    finally:
        writer.release()
    cfg = load_config()
    cfg["video"].update(allow_constant_fps_timestamps=True, max_width_px=48)
    calls = []

    def detector(rgb, timestamp_ms):
        calls.append((rgb.shape, timestamp_ms))
        return FrameDetection(np.tile([.5, .5, 0.0], (21, 1)))

    recording = extract_landmarks(path, "hand", cfg, detector=detector)
    assert len(calls) == 60
    assert calls[0][0] == (32, 48, 3)
    assert (recording.width, recording.height) == (48, 32)
    assert recording.metadata["decoded_width_after_rotation"] == 96
    assert recording.metadata["decoded_height_after_rotation"] == 64
    assert recording.t[-1] == pytest.approx(59 / 30)
    assert recording.confidence is None
    assert recording.diagnostics["subsampled"] is False
    assert recording.metadata["timestamp_source"] == "frame_index_assuming_constant_fps"
