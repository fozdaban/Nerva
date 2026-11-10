# Postural sway: acceleration-domain proxies

`extract_sway(recording, config)` supports configured trunk placements such as
lower back or chest. Horizontal axes must be specified in `features.sway` and,
by default, metadata must contain `horizontal_alignment_confirmed: true`.
This confirms an external mounting convention; no automatic orientation or
gravity-alignment estimate is implemented. Wrist/ankle input returns `retest`.

Only acceleration is required. Set the adapter's explicit
`allow_missing_gyroscope=true` for acceleration-only exports; absent gyro is not
filled with zeros. Sway quality checks still require all three acceleration axes,
even though features use the configured horizontal pair.

## Preprocessing and feature-stage result

Finite unclipped acceleration segments are interpolated at the configured native
rate, then anti-aliased/resampled to the target rate. Missing rows and long gaps
break segments; they are never spanned. Constant axis offsets are removed before
independent 0.1–3 Hz Butterworth filtering. This task does **not** apply the shared
0.3 Hz high-pass, which would remove low-frequency sway components. Filters use
zero-phase second-order sections. Default trimming is 3 seconds at each segment
edge, followed by complete 4-second, 50%-overlapping windows.

At least three windows and 50% of expected windows must survive, otherwise the
result is the existing minimal `retest` response without features. Quiet stillness
can pass: zero motion is valid and does not imply any clinical conclusion.

Successful output follows the internal feature-stage pattern: named features,
window values, IQRs, window counts, quality score and limitations. These are not
new fields in the proposed mobile API. Feature values are window medians except
global path, which sums each unique acceleration edge covered by windows once.
Edges are deduplicated by uninterrupted segment and sample offset; no synthetic
edge crosses a gap. `observed_duration_s` is the duration represented by those
unique edges. Its path total is not extrapolated to trimmed/excluded time.
Overlapping-window IQRs are descriptive spread, not 95% confidence intervals.
The path IQR describes window paths, not uncertainty in the global path total.

| Feature | Calculation and meaning | Units |
| --- | --- | --- |
| `horizontal_acceleration_rms_m_s2` | sqrt(mean(x²+y²)) after per-window centering; horizontal acceleration variation | m/s² |
| `acceleration_path_length_proxy_m_s2` | Sum of unique Euclidean differences between successive horizontal acceleration samples covered by accepted windows; per-window values use only that window | m/s² |
| `mean_frequency_hz` | Power-weighted centroid of summed horizontal Welch spectra, with DC excluded | Hz; null for zero power |
| `acceleration_ellipse_area_95_m2_s4` | π × chi-square(2,0.95) × sqrt(det(sample covariance)) | (m/s²)² |

The ellipse is a **nominal Gaussian covariance contour**, not a confidence
interval or guaranteed 95% containment of a non-Gaussian trajectory. A line or
constant trajectory has zero area. Welch uses a Hann window and the full feature
window. Four-second windows do not resolve very slow components well; do not
interpret the mean frequency as a precise low-frequency oscillation estimate.

## Assumptions and limitations

These are acceleration-domain measures, **not displacement, physical body sway
path, or center-of-pressure area**. No double integration is performed. Mounting,
tilt and gravity projection can change the values without changing actual body
translation. The path proxy depends on sample rate, filters and window duration;
compare sessions only with consistent settings. Slow filters have long edge
transients; configurable trimming reduces but does not prove their absence.

No population reference ranges, clinical significance thresholds or real-person
validity are established. Synthetic tests check known ellipse RMS/area/frequency,
circular acceleration-path length, horizontal rotations, acceleration-only input,
stillness, excluded gaps and quality rejection.
