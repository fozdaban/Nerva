# IMU quality gate

`assess_imu(recording, task=None, config=None)` evaluates a canonical `Recording`.
Pass the same config used at ingestion. Task defaults to metadata `task_name`;
`task_mapping` maps guided-test names to configurable task families. Unknown
tasks raise `ValueError`; they never silently receive permissive defaults.

```python
from motor_pipeline.io import normalize_imu
from motor_pipeline.synthetic import add_artifact, tremor
from motor_pipeline.quality import assess_imu

recording = normalize_imu(add_artifact(tremor(), "gap", duration_s=1))
report = assess_imu(recording, "rest_tremor")
assert report.status == "retest"
print(report.as_dict())          # Existing quality object: score and reasons.
print(report.retest_response())  # {"status": "retest", "reasons": [...]}
```

`IMUQuality.measurements` holds internal audit evidence for debugging and tests;
it is not an addition to the proposed task JSON schema. No motor feature is
computed by the quality gate. Successful checks return status `ok` and
`retest_response()` returns None. The caller must stop feature extraction on
`retest`; full task API orchestration is a later stage.

## Configurable requirements and limits

These are provisional engineering defaults, not clinically validated thresholds.
All settings below live under `quality` in `config.yaml`, except hardware ranges
under `input` and task aliases under `task_mapping`.

| Check | Default | Calculation / interpretation |
| --- | --- | --- |
| Minimum score | 0.75 | Retest when score is lower |
| Required channels | All six for tremor, gait, IMU tapping, reps; acceleration only for sway | Missing column or entirely nonfinite required channel forces score 0 |
| Minimum duration | Tremor/tapping/reps 10 s, gait 15 s, sway 20 s | Last minus first timestamp plus one nominal sample period |
| Missing data | At most 5% | Fraction of expected time slots with any required signal absent |
| Timestamp gap definition | Greater than 2 nominal periods | Gap count and longest gap are retained in audit evidence |
| Longest timestamp gap | At most 0.25 s | Forces retest even with a small total missing fraction |
| Clipping threshold | At least 98% of sensor full scale | Absolute per-axis value; a row counts once if any required axis clips |
| Clipped rows | At most 1% | Full scale defaults to 8 × 9.80665 m/s² and 1000 °/s |
| Implausible threshold | Greater than 105% of full scale | Any such row forces retest by default |
| Duplicates before repair | At most 1% of original samples | Uses adapter audit, so deduplication cannot conceal the loss |
| Backward transitions before repair | At most 1% of original samples | Uses adapter audit; not a clock-reset detector |
| Rate disagreement | At most 10% | Absolute relative error between inverse median sample interval and configured rate |
| Movement RMS | At least 0.02 m/s² for active tasks | Mean-centered acceleration vector RMS; enabled for gait, IMU tapping and reps |

Limits allow equality (with a small floating-point tolerance). Rate and sensor
limits use metadata/config, not inferred hardware specifications. Stationary
tremor and sway recordings do not need to exceed a movement threshold: little
movement is a valid observation for those tasks. Presence of movement does not
establish that the requested task was performed correctly.

## Missing data arithmetic

For every positive interval, inferred lost slots are
`max(round(interval_seconds * nominal_rate) - 1, 0)`. Single dropped samples are
counted even though their two-period interval is not a flagged >2-period gap.
The missing fraction is:

```
(inferred_lost_slots + retained_rows_missing_any_required_channel)
/ (inferred_lost_slots + retained_rows)
```

This is a union over required channels, not an average over six channels. It
does not double count missing values in the same row. This quality estimate is
deliberately more sensitive to single-slot loss than the adapter's diagnostic
`estimated_missing_time_fraction`, which only counts intervals above its gap
threshold. Irregular clocks or large jitter can make slot estimates inaccurate.
Dropped samples before the first or after the last timestamp are unknowable
without assessment start/end markers.

## Score formula and reasons

Take the minimum of:

- 1 (a ceiling);
- duration / required duration (capped at 1);
- 1 minus each of missing, clipped, implausible, duplicate and backward-transition fractions;
- 1 minus relative sampling-rate error (floored at 0);
- measured movement RMS / minimum RMS (capped at 1), when movement is expected.

If any individual requirement fails, cap this score at half `minimum_score`.
This makes every failed gate fall below the configured acceptance threshold.
An otherwise valid recording with a score below the cutoff receives a combined
quality reason. Scores are deterministic heuristics; a score of 0.9 does not
mean a 90% chance of accurate motor measurements. Reasons describe failed
requirements; accepted small imperfections remain in audit measurements.

## Boundaries and known limitations

- Checks use retained sensor rows and the original adapter timestamp audit.
  Sensor values on rows discarded during duplicate repair cannot be audited.
- Missing/nonfinite values remain unchanged. A pass permits small missingness;
  later preprocessing must define interpolation and window-exclusion policies.
- Flat acceleration during an active task is rejected. A stuck sensor during a
  stationary task cannot be distinguished from real stillness by this test alone.
- Mean subtraction removes a constant gravity/offset vector. It does not track
  gravity during rotations or prove task adherence. Orientation changes and
  subrange sudden movements can pass; task-specific artifact checks remain needed.
- Movement RMS is skipped when values exceed the implausible threshold, avoiding
  overflow from corrupt sensor values. Those values already require rejection
  with default settings. Do not relax that gate to accept corrupt data.
- Gaps in finite timestamps are gated separately from the missing fraction;
  contiguous NaN runs currently contribute to the fraction, without their own
  maximum-run-duration gate.
- Dataframes are assumed numeric, as guaranteed by the input adapter. Invalid
  timestamps in a manually constructed canonical recording return retest; bad
  configuration raises rather than disguising a programming error as poor data.
- No landmark, camera, clinical, ICC, SEM or real-world validation is included.
