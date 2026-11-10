# Video landmark and feature pipeline

Implemented functions are internal analysis stages, not changes to the proposed
mobile JSON schema. There is no diagnostic classification or model training.

## Input and extraction

```python
from motor_pipeline.config import load_config
from motor_pipeline.vision import extract_landmarks
from motor_pipeline.features import extract_tapping_video, extract_gait_video

config = load_config()
config["video"]["hand_model_path"] = "models/hand_landmarker.task"
# Caller supplies the model locally; the pipeline never downloads one.
landmarks = extract_landmarks("phone_tapping.mp4", "hand", config)
result = extract_tapping_video(landmarks, config)
```

Pose extraction uses `kind="pose"` and `video.pose_model_path`; pass its result to
`extract_gait_video`. OpenCV and MediaPipe are lazy optional dependencies and
not needed to analyze already-extracted landmark arrays. The optional `video`
dependency group contains only packages in the requested stack. Verify the
model asset's license and redistribution terms before distributing it.

The adapter uses MediaPipe Tasks in VIDEO mode with increasing millisecond
timestamps and an explicit local model asset. It asks for up to two objects and
treats zero/multiple detected hands or people as a missing frame rather than
silently choosing an object. This does not prove persistent identity when only
one hand/person is visible. Guided recordings must contain one intended hand or
participant. The implementation follows the official
[Hand Landmarker Python example](https://developers.google.com/edge/mediapipe/solutions/vision/hand_landmarker?tab=Python)
and [Pose Landmarker Python example](https://developers.google.com/edge/mediapipe/solutions/vision/pose_landmarker?tab=Python).

MediaPipe provides hand coordinates and handedness but not a per-frame hand
detection-confidence output. That mean is null. A handedness classification
score is never reinterpreted as landmark certainty. Pose supplies landmark
visibility/presence; the adapter uses the smaller available value per landmark.
Detection/presence/tracking threshold controls are configured separately under
`video`; their values are not reported as measured confidence.

OpenCV reads each frame without implicit subsampling. Decoder presentation
timestamps must be finite and increasing. `allow_constant_fps_timestamps=true`
explicitly opts into frame-index/fps timing and records that assumption; it is
not safe for unknown variable-rate recordings. Rotation is explicitly configured;
confirm the decoder's orientation handling before applying an extra rotation.
Frames can be resized to `max_width_px` with preserved aspect ratio. Returned
dimensions are the actual processed dimensions, including rounding, so metric
geometry respects the image fed to the detector.

Exceeding `max_frames` raises `VideoInputError`; no partial recording is silently
accepted. Clearly early decode termination relative to a positive advertised
frame count is also rejected (a one-frame tolerance accommodates metadata
rounding). Unknown or inaccurate length metadata cannot guarantee completeness.
File/model/setup errors are raised; a future task API must translate them into
invalid-input/retest responses. Capture and owned detector resources are closed.

## Canonical landmark evidence

`LandmarkRecording` contains:

- `t`: float seconds, one value per decoded frame;
- `points`: normalized xyz array, shape `(frames, 21, 3)` for hands or
  `(frames, 33, 3)` for pose; missed frames are NaN;
- `width`, `height`: actual image dimensions after configured transforms;
- `confidence`: per-landmark visibility/presence array, or None when unavailable;
- metadata with capture fps, timestamp source and explicit rotation;
- diagnostics with decoded/advertised counts and `subsampled=false`.

`image_xy()` multiplies normalized x by width/height. Distances therefore use
isotropic frame-height units rather than incorrectly treating normalized x/y as
equivalent lengths. Features are image-plane measurements; z is retained as
evidence but not treated as calibrated metric depth.

## Video quality

`assess_video` measures tracked fraction, usable fraction, mean available
visibility/presence, actual processed fps, duration, longest timestamp gap and
required-landmark out-of-frame fraction. Tracked fraction's denominator includes
all decoded frames, not just successful detections. Capture fps remains separate
from processed fps. There is no hidden frame skipping.

Default gates require 80% usable frames, 20 fps, task-specific duration (10 s
tapping, 15 s gait), reported pose landmark confidence ≥0.6, timestamp gaps no
longer than 0.2 s, and at most 10% cropped frames. Thresholds are configurable.
Hand confidence is unavailable and is not made into a fabricated rejection or
perfect-confidence score. Pose confidence is required because its task provides
that evidence. Crop checks apply to task-relevant landmarks; unused landmarks
cannot invalidate a task, while required fingertips/ankles/wrists cannot leave
the image unnoticed.

The score is a deterministic minimum of usable fraction, fps sufficiency and
duration sufficiency. Any failed gate caps it below the configured cutoff.
It is not a probability of accurate measurement. Rejections return `retest`
with reasons and no metrics. Missing/cropped/low-visibility frames and gaps
also split feature segments; they are never interpolated into events.

## Video tapping metrics

Normalize thumb-tip/index-tip 2D distance (4↔8) by wrist/middle-MCP distance
(0↔9). Defaults use a hand length ≥0.03 frame heights and opening-peak
prominence ≥0.08 hand lengths, with at least 0.15 s between openings. Peak times
use three-sample parabolic refinement to reduce frame-quantization jitter.
Amplitude uses the larger observed peak-to-side-base range: a task beginning
mid-cycle does not force half-prominence into a full amplitude trend.

| Feature | Calculation | Unit |
| --- | --- | --- |
| `tap_count` | Unique opening events, deduplicated by source frame across overlapping windows | count |
| `tap_rate_hz` | Inverse mean within-segment interval | taps/s |
| `inter_tap_interval_cv` | Sample SD / mean interval | fraction |
| `median_tap_amplitude_hand_lengths` | Median event opening amplitude relative to hand size | hand lengths |
| `amplitude_slope_hand_lengths_per_s` | Least-squares event amplitude slope against elapsed time | hand lengths/s |
| `amplitude_decrement_fraction` | 1 − fitted last / fitted first amplitude over observed span | signed fraction |
| `hesitation_count` | Unique intervals >2× their median, configurable | count |
| `tracked_frame_fraction` | Fraction of all decoded frames with required coordinates tracked | fraction |

Global slope/decrement are withheld across discontinuous segments. Clean data
with insufficient openings still requests a retest rather than inventing zero
rhythm metrics. A maximum opening is a tapping-cycle proxy, not proof of contact.
Perspective, occlusion, out-of-plane motion, landmark noise and hand-size changes
can bias amplitude. No normal/clinical range or clinical accuracy is established.

## Video gait metrics

Use hip-relative ankle image trajectories along a configurable x/y axis (default
y), normalized by a shoulder-to-hip scale of at least 0.05 frame heights.
This configurable size gate avoids amplifying negligible torso distances into
large normalized movements. Peak separation defaults to 0.3 s
per ankle and prominence to 0.02 body lengths. Alternate ankle events are needed
to support step timing; unilateral activity or coincident ankle peaks are
ambiguous. Event counts deduplicate source-frame/ankle detections across windows.
Intervals cannot cross tracking segments or unsupported-motion holes.

| Feature | Calculation | Unit |
| --- | --- | --- |
| `step_count` | Unique accepted ankle-trajectory peak events | count |
| `cadence_steps_min` | 60 / mean included alternating-ankle interval | steps/min |
| `mean_step_time_s` | Mean included event interval | s |
| `step_time_cv` | Sample SD / mean included interval | fraction |
| `arm_swing_amplitude_body_lengths` | Median window/wrist norm of 95th–5th percentile hip-relative xy range, normalized by body scale | body lengths |
| `tracked_frame_fraction` | Required-coordinate tracking fraction over all decoded frames | fraction |

These are projected motion/event proxies, not validated heel strikes, physical
speed/distance, or anatomical joint angles. Camera view and occlusion strongly
affect them. Bilateral tracking does not establish clinical asymmetry validity.

Both feature functions require three supported complete windows and 50% of
expected windows by default. They retain per-window values, event times, segment
IDs and IQR evidence. IQRs describe dependent window estimates; they are not
calibrated confidence intervals, and per-window count spread is not uncertainty
in the global deduplicated count.

## Executed validation and remaining work

41 tests pass on deterministic landmarks and adapter fixtures: tapping rates
1.5/2/3 Hz at 24/30/60 fps; known decrement; aspect correction; unique counts;
gait cadence 80/120/160; simultaneous/unilateral ankle rejection; unsupported
motion holes; missing/low-confidence/cropped frames; strict/CFR timestamps;
maximum/incomplete-frame rejection; JSON-safe outputs; and resource release.
One integration test writes/reads a real synthetic 60-frame AVI through OpenCV
with an injected detector, validating rotation, resize geometry and timestamps.

MediaPipe is not installed locally, no model asset was downloaded, and real
MediaPipe inference or a human phone clip has **not** been validated. The Tasks
adapter is checked against official examples and result-shape mocks only.
Model-package compatibility, model licensing, real clip accuracy, full landmark
runtime and the under-one-second recording target remain unverified. Frame-by-
frame landmark extraction can dominate compute even when feature calculations
are small. Benchmark decoding and inference separately before deployment.
