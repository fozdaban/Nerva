# Stage 1 validation

Date: 2026-11-10. Scope: IMU ingestion and synthetic fixtures only.

## Executed checks

| Module | Result | Evidence |
| --- | --- | --- |
| Input adapter | 32 tests passed | 50/100/200 Hz; timestamp units and timezone-aware ISO; axis/unit conversion; gaps and duplicates; repair/rejection; JSON packets; CSV packet anchors; phyphox headers; missing gyro; metadata precedence |
| Synthetic fixtures | 30 tests passed | Known tremor frequencies/amplitudes; seeded noise and gravity; tapping rate/decrement; one-limb gait cadence; six artifact types; invalid parameters |

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

## Ingestion timing

On this workstation, 20 calls to `normalize_imu` on the same in-memory
30-second, 100 Hz fixture (3,000 rows) took median **10.45 ms**, maximum
**14.46 ms**, measured with `time.perf_counter`. This includes normalization
and diagnostics, excludes disk I/O, and is a small local smoke benchmark.
No portable latency guarantee follows from this result. Future task analysis,
video decoding and landmark extraction need separate benchmarks. Video runtime
will depend on frame count, resolution and decoding; the full recording target
of well under a second is not yet demonstrated.

## What the fixtures establish

The tremor fixture has a known signed single-axis oscillation; its Fourier peak
is recovered within 0.05 Hz. This tests the generator, not a future Welch feature
implementation or accuracy in human recordings. A magnitude of a signed
single-axis oscillation can double its frequency. Future tremor extraction must
test this explicitly and preserve axis-level evidence before combining power.

The walking fixture produces one positive angular-velocity peak per **stride**
of the instrumented ankle. Cadence is twice that peak rate. It does not establish
that a wrist peak corresponds to a step, or validate turn or walking-speed
proxies. Tapping fixtures are smooth positive pulses with an explicitly known
first-to-last amplitude reduction. They omit realistic wrist/finger coupling.

## Remaining validation and failure cases

- No motor features, quality score, task JSON serializer or retest policy has
  been implemented. Ingestion diagnostics are evidence for the next module.
- A missing required channel or timestamp raises an input error. Optional missing
  gyroscope values remain NaN; later tasks must reject absent required signals.
- Jitter, clipping and sudden movement are retained. Clipping artifacts are
  generated but not scored at this stage. Short recordings can be loaded and
  must be rejected later according to task-specific duration requirements.
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
