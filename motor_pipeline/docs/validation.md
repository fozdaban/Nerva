# Validation through IMU tapping features

Date: 2026-11-10. Scope: IMU ingestion, synthetic fixtures, IMU quality gates,
preprocessing, tremor, gait and IMU tapping features.

## Executed checks

| Module | Result | Evidence |
| --- | --- | --- |
| Input adapter | 32 tests passed | 50/100/200 Hz; timestamp units and timezone-aware ISO; axis/unit conversion; gaps and duplicates; repair/rejection; JSON packets; CSV packet anchors; phyphox headers; missing gyro; metadata precedence |
| Synthetic fixtures | 30 tests passed | Known tremor frequencies/amplitudes; seeded noise and gravity; tapping rate/decrement; one-limb gait cadence; six artifact types; invalid parameters |
| IMU quality | 50 tests passed | Clean signals at 50/100/200 Hz; stationary/active task rules; clipping and implausible values; gaps, required-channel missingness and repair audits; duration boundaries; rate disagreement; configurable thresholds; retest JSON without metrics |
| Preprocessing | 16 tests passed | Resampling and RMS at multiple rates; gravity removal; anti-alias filtering; independent segments around gaps/NaNs/clipping; high-order filter stability; invalid configuration |
| Tremor | 44 tests passed | ±0.2 Hz frequency recovery at 3, 3.2, 5, 5.1, 7, 11.8 and 12 Hz across 50/100/200 Hz; power/RMS recovery; rotation invariance; magnitude doubling; low-amplitude abstention; intermittent windows; fragmented signals; JSON-safe values |
| Gait | 39 tests passed | Cadence at 80/120/160 steps/min across 50/100/200 Hz; stride time/CV; unique counts; variable intervals; fixed rotations; signed axes; wrist swing amplitude and explicit stride mapping; calibration-based speed; exclusions/retest; JSON-safe evidence |
| IMU tapping | 38 tests passed | Tap rates 1.5/2/3/4 Hz across 50/100/200 Hz; observed-span decrement; increasing amplitudes; unique counts; long pause vs missing data; polarity/rotation; gyro units; fragmented/absent signals; JSON-safe output |

Tests ran on Windows with CPython 3.13.3, NumPy 2.5.4, pandas 2.3.0 and
pytest 9.2.0. Python 3.11 is the requested target and the source uses compatible
syntax, but that interpreter was unavailable locally. Python 3.11 execution
remains unverified. Pytest was installed into ignored `.test-deps` for this
session; NumPy/pandas were already available. Pytest plugin autoload was disabled
to isolate these tests from unrelated installed packages.

Normal setup and verification on a machine with Python 3.11:

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\python -m pip install -e ".[test]"
.\.venv\Scripts\python -m pytest -q
```

In this session, tests used `PYTHONPATH=.test-deps` and
`PYTEST_DISABLE_PLUGIN_AUTOLOAD=1`. Installation created Windows permissions
that required running pytest outside the restricted sandbox. These environment
workarounds are not application requirements.

SciPy 1.17.1 was already available and is now declared as a runtime dependency
for preprocessing and Welch features, as authorized in the requested stack.

## Ingestion timing

On this workstation, 20 calls to `normalize_imu` on the same in-memory
30-second, 100 Hz fixture (3,000 rows) took median **10.45 ms**, maximum
**14.46 ms**, measured with `time.perf_counter`. This includes normalization
and diagnostics, excludes disk I/O, and is a small local smoke benchmark.
No portable latency guarantee follows from this result. Future task analysis,
video decoding and landmark extraction need separate benchmarks. Video runtime
will depend on frame count, resolution and decoding; the full recording target
of well under a second is not yet demonstrated.

The complete tremor feature call (quality, preprocessing, both sensor groups,
magnitude spectra, window features and IQRs) on an already-normalized 30-second,
100 Hz recording took median **25.71 ms**, maximum **27.04 ms**, across 20 local
calls, producing 13 complete windows. Disk loading, normalization, process
startup and library import time are excluded. This is a local steady-state
smoke benchmark, not a phone/server latency guarantee.

Gait analysis (quality, preprocessing, projection, peak detection, event
reconciliation and window spread) on an already-normalized 30-second, 100 Hz
ankle fixture took median **11.78 ms**, maximum **21.34 ms**, across 20 local
calls, producing 13 accepted windows. This excludes disk loading, ingestion,
process startup and imports. This is another local steady-state smoke benchmark.

IMU tapping on an already-normalized 30-second, 100 Hz fixture took median
**8.65 ms**, maximum **9.56 ms**, over 20 calls with 13 supported windows.
This includes quality, preprocessing, event merging, timing, amplitude fitting
and IQRs; it excludes I/O, ingestion and startup, as above.

## What the fixtures establish

The tremor fixture has a known signed single-axis oscillation; its Fourier peak
is recovered within 0.05 Hz by the fixture tests. Separate Welch feature tests
recover configured frequencies within ±0.2 Hz, including search-band boundaries.
The implemented spectral trace sums axis powers, preserving the original
frequency and invariance to a fixed sensor rotation. Supplemental magnitude
powers are labeled separately because magnitude rectification can double a
sinusoid's frequency. This does not establish accuracy in human recordings.

Band-edge tests initially found unstable window decisions at a 50% power-ratio
cutoff: a boundary peak puts about half its power outside the integration band.
The provisional default was changed to 40%, retaining amplitude and
peak/background checks. Clean seeded noise and low oscillations return null
frequency; artifact/fragmentation cases return retest without features. Window
spread is IQR, not a calibrated 95% confidence interval. Thresholds require
real-data evaluation.

The walking fixture produces one positive angular-velocity peak per **stride**
of the instrumented ankle. Cadence is twice that peak rate. It does not establish
that a wrist peak corresponds to a step, or validate turn or walking-speed
proxies. Tapping fixtures are smooth positive pulses with an explicitly known
first-to-last amplitude reduction. They omit realistic wrist/finger coupling.

Gait tests recover configured cadence within 2 steps/min and mean stride time
within 0.02 s. An alternating 0.85/1.15 s fixture recovers CV near 0.15 rather
than flattening variability away. A 30-second, 120 steps/min fixture yields
27 complete supported strides after edge/window exclusions; summing overlapping
windows would overcount. Small missing intervals are not converted to strides.
Wrist integration recovers the analytical sinusoid angle proxy A/(πf) within 4%,
but this is a math-fixture check, not anatomical or clinical validation.

Wrist cadence/stride metrics are withheld unless one swing cycle per stride is
explicitly enabled. Speed is null unless a constant stride length is configured.
Turn count is null. These limits are tested, including the one-sensor asymmetry
limitation. Real walking, turns, pauses, irregular gait, strap movement and
arm-swing/stride coupling require recorded-data validation. Periodic gestures
can satisfy these engineering cycle checks without being walking.

Tapping tests recover configured rates within 2%, constant-rhythm CV below 0.05,
and observed-span decrement within 0.03 absolute. Decrement is compared against
the known envelope evaluated at retained event times, not the full source span.
A 3-second removed movement run creates one 3.5-second inter-tap hesitation in a
2 Hz fixture. Missing sensor data instead separates segments and is not counted
as hesitation. Global amplitude trends across those segments are withheld.
These checks verify fixture behavior, not correspondence between wrist motion
and actual finger taps.

## Remaining validation and failure cases

- Tremor/gait/IMU tapping features, quality scores, reasons and retest responses
  are implemented and tested. Video/sway/reps, full task JSON serialization,
  calibrated confidence and summaries remain unimplemented.
- A missing required channel or timestamp raises an input error. Optional missing
  gyroscope values remain NaN; later tasks must reject absent required signals.
- Jitter, clipping and sudden movement are retained at ingestion. The quality
  module rejects excessive clipping, implausible values and recordings below
  task-specific durations. Subrange sudden movements can still pass; there is
  no task-specific movement contamination detector yet.
- Sorting is suitable for packet reordering, but not clock resets. Use rejection
  until sensor timestamp semantics are confirmed. CSV packet boundary ambiguity
  and configured-rate interpolation are described in the data contract.
- Low amplitude, motion contamination, irregular tapping, missing video frames
  and camera perspective need task-level error analysis and real recordings.
- ICC, SEM, MDC95, longitudinal change thresholds and subject-split leakage
  audits need repeated participant recordings or labeled datasets. None has
  been estimated from these synthetic fixtures.
- No public data was downloaded or used. PADS/mPower license, access, placement,
  labels and rate review is deferred to the requested dataset/model stage.
- No clinical accuracy, typical clinical ranges, diagnostic validity or
  generalization claim is supported by these tests.
