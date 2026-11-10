# Exercise repetitions from IMU

`extract_reps(recording, config)` provides internal named features, window
estimates, spread and complete-cycle evidence. It is not a diagnosis, exercise
form assessment or the final mobile API. It requires a placement listed under
`features.reps.accepted_placements` and the configured IMU quality requirements.

## Method

Use the shared complete-segment resampling/filtering pipeline, with the
exercise-specific 0.03 Hz gravity high-pass and 0.1–5 Hz band-pass defaults.
These keep the slow exercise band separate from the 0.3 Hz tremor default.
The task setting is applied to a config copy, so other tasks are unchanged.
Default primary sensor is gyro; acceleration can be selected. Project onto a
configured signed axis or the highest-variance PCA direction. Auto sign is
deterministic; it is not an anatomical phase calibration. Do not rectify the
signal and thereby count both positive and negative lobes as separate reps.

Stitch only the union of motion-supported windows within each original prepared
segment, then detect positive peaks once per continuous analysis run. A hole in
that accepted union splits a new run even without missing raw samples. Rejected
windows cannot contribute cycles or apparent slowing. Shared overlapping
samples are not new events. Default minimum separation is 0.4 s, prominence is
at least max(5 °/s, 20% of the segment peak-to-peak signal) for gyro or
max(0.1 m/s², 20%) for acceleration. Supported windows require projected RMS
at least 2 °/s or 0.03 m/s², with at least three such windows and 50% of expected
windows. Require at least two complete peak-to-peak cycles in total. Defaults
accept cycle durations of 0.4–8 s; out-of-range detected intervals request retest
rather than silently disappearing from variability estimates.

These are engineering defaults, not clinically validated thresholds. Repeated
gestures, higher harmonics or a strong unrelated axis can produce periodic peaks.
A known task-specific axis/mounting is preferable when available.

## Metrics

| Feature | Calculation | Units |
| --- | --- | --- |
| `rep_count` | Number of complete same-phase peak-to-peak cycles; partial leading/trailing cycles excluded | integer |
| `mean_rep_duration_s` | Mean of complete within-segment durations | s |
| `rep_duration_cv` | Sample duration SD / mean | fraction |
| `range_of_motion_proxy` | Median per-cycle signal excursion, as described below | degrees for gyro; m/s² for acceleration |
| `range_of_motion_cv` | Sample SD / mean of cycle excursions | fraction |
| `duration_slope_s_per_s` | Linear slope of rep duration versus cycle midpoint time | s/s |
| `slowing_fraction` | Fitted last duration / fitted first duration − 1 over observed cycles | signed fraction |

For gyro, trapezoidally integrate projected angular velocity within a complete
cycle and subtract a linear term that closes the cycle; report the integrated
angle peak-to-peak range. This is a dominant-axis angle proxy, not anatomical
joint range. For acceleration, use projected acceleration peak-to-peak and
explicitly retain acceleration units; no distance is reconstructed.

Positive duration slope means cycles took longer over the observed interior set.
Speeding-up remains negative; values are not forced into slowing. The first and
last complete cycle of a continuous analysis run are excluded from the global
trend fit (`trend_boundary_cycles=1`); require at least three remaining interior
cycles (`minimum_trend_cycles=3`). Thus at least five total complete cycles are
needed for a global trend. Short slow tasks can return cycle timing/counts with
null slowing rather than treating filter-edge timing bias as a real trend.
Slowing fraction describes the retained fit span, not the task's excluded edges.
Global trends are withheld across discontinuous segments or analysis runs.
Missing or unsupported samples never become a long
repetition. Counts cover observed complete cycles and do not extrapolate over
missing or trimmed time. With peak-to-peak counting, a task with N detected
same-phase peaks contains N−1 complete cycles per uninterrupted segment.

`per_window` keeps estimates for complete cycles inside each window. Global
counts are not sums of overlapping window counts. `spread` records window IQR,
window-value count and recording quality score. These are descriptive IQRs, not
95% confidence intervals for total counts or slopes. A window can have no complete
cycle and hence a zero count/null duration even when a long cycle is supported
by the stitched segment. Global event times and cycle boundaries remain available
for auditing. No real exercise, camera agreement or clinical precision has been
validated from the synthetic checks.

The fixed 0.5-second preprocessing trim does not prove absence of low-frequency
filter transients. Boundary-cycle exclusion reduces bias but does not establish
clinical trend precision, especially with slow cycles or short recordings. Tests
include constant 4-second/20-second and 6-second/30-second tasks whose global
trends are deliberately withheld, plus longer constant tasks to check for large
spurious slopes. Filtered angle proxies can retain boundary amplitude bias.
Duration bounds remain conservative: a true cycle exactly at the configured
upper limit can be shifted beyond that limit by filtering and request retest.
Do not interpret this as evidence of a physical timing violation.
