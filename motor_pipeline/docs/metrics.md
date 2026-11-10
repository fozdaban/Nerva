# Implemented motor features: tremor, gait and IMU tapping

These are motor signal measurements and provisional signal-detection rules.
There are no clinical reference ranges or diagnostic interpretations. All bands,
thresholds and window settings are configured in `config.yaml`.

## Processing and evidence

`extract_tremor(recording, config)` first applies task-aware IMU quality checks.
It returns the existing minimal `retest` response if input quality is inadequate.
It also returns `retest` when preprocessing leaves fewer than three complete
windows or fewer than 50% of the expected windows (configurable).

Accepted recordings return a **feature-stage dictionary** with `status`, `quality`,
`features`, `per_window`, `spread`, `n_valid_windows` and `expected_windows`.
This is internal analysis evidence, not the future app API envelope; the proposed
task JSON schema is unchanged. Final confidence levels, personal changes and
patient/clinician summaries are later stages.

Measurements are aggregated by median, except presence indicators, which are
averaged to obtain the fraction of windows meeting the rule. `spread` contains
each metric's 25th/75th percentiles, supporting window count and recording quality
score. A frequency uses only windows meeting the rule; unavailable frequencies
are null. Frequency SD is null with fewer than two supporting windows. Its IQR
is null because it is already a cross-window statistic. The IQRs of other features
describe overlapping, dependent windows: **they are not 95% confidence intervals**.
`per_window` retains relative start time and every window-level feature.

## Preprocessing

1. Use canonical m/s² and °/s. All six axes are required for these dual-sensor
   features, regardless of the configured primary sensor.
2. Break segments at nonfinite or clipped rows and long timestamp gaps.
   Interpolate timestamp jitter on each segment's nominal source grid. The
   maximum gap is 0.025 s, or 1.5 source periods if larger. A row with NaN breaks
   a segment; it is not imputed. A dropped sample can be interpolated if its
   resulting timestamp gap is within this limit.
3. Remove constant axis means, then use polyphase resampling with anti-alias
   filtering to 100 Hz by default. Do not stretch across lost segments.
4. Apply a 0.3 Hz high-pass Butterworth filter for slowly varying gravity/offset.
   This high-passed signal supplies the broadband spectral denominator.
5. Separately band-pass 2–15 Hz for RMS amplitude. Filters default to order 4,
   using SciPy `sosfiltfilt`, a numerically stable forward/backward zero-phase
   implementation with second-order sections.
6. Trim 0.5 s at both ends of every segment. Take complete 4-second windows
   with 50% overlap; discard short segments and partial final windows.

Trimming reduces filter transients but does not guarantee their absence. Higher
orders and slow signals may require longer trims. Gravity removal cannot recover
a changing sensor orientation or distinguish intentional motion from oscillation.

## Spectral convention

Compute one-sided Welch PSD with a Hann window, constant detrending, density
scaling, and 4-second segments (capped at the feature window), with 50% Welch
overlap when multiple segments fit. Default frequency bins are 0.25 Hz apart;
each default feature window contains one Welch segment. Integrate with trapezoids
and interpolated band boundaries.

Sum axis PSDs to obtain a spectral trace. The trace and vector RMS are invariant
to a **fixed** orthogonal rotation. PSD of a Euclidean magnitude can rectify a
sinusoid and introduce a doubled frequency, so magnitude spectra are supplemental
and never determine the primary frequency. Refine the dominant trace bin with
three-bin log-power interpolation, constrained to the search band. Interpolation
does not create additional spectral resolution.

## Named features

`{sensor}` expands to `accel` and `gyro`. Acceleration power units are `(m/s²)²`,
gyro power units are `(°/s)²`; RMS uses m/s² and °/s respectively.

| Name | Calculation and plain-language meaning | Unit / mathematical range |
| --- | --- | --- |
| `{sensor}_dominant_freq_hz` | Strongest refined trace peak in 3–12 Hz, only when the signal rule is met | Hz; configured search band, or null |
| `{sensor}_power_narrow` | Trace-PSD integral in 3–7 Hz; narrower-band oscillation energy | Sensor power; ≥0 |
| `{sensor}_power_band` | Trace-PSD integral in 3–12 Hz; search-band oscillation energy | Sensor power; ≥0 |
| `{sensor}_band_power_ratio` | Search-band power / total high-passed trace power, 0–Nyquist | Dimensionless; 0–1 |
| `{sensor}_rms` | sqrt(mean(x²+y²+z²)) of the 2–15 Hz filtered axes; combined amplitude | Sensor RMS; ≥0 |
| `{sensor}_magnitude_power_band` | High-passed vector magnitude's PSD integral in 3–12 Hz; supplemental rectified-signal power | Sensor power; ≥0 |
| `{sensor}_magnitude_band_power_ratio` | Magnitude search-band power / total magnitude power after Welch constant detrending | Dimensionless; 0–1 |
| `{sensor}_tremor_present` | Per-window 0/1 rule indicator; aggregate is its mean | Fraction; 0–1 |
| `{sensor}_peak_frequency_sd_hz` | Sample SD of accepted window peak frequencies; smaller means more consistent peaks | Hz; ≥0, or null |
| `dominant_freq_hz` | Alias for the primary sensor's dominant frequency; gyro by default | Hz, or null |
| `tremor_present_fraction` | Alias for the primary sensor's mean window indicator | Fraction; 0–1 |

There are no established population-based typical ranges. A null frequency
must not be interpreted as absence of a disease or symptom.

## Provisional window rule

All three conditions must hold for each sensor:

- sqrt(3–12 Hz power) ≥ 0.03 m/s² for acceleration, or 0.5 °/s for gyro;
- 3–12 Hz power / total dynamic power ≥ 0.40;
- strongest in-band PSD bin / median out-of-band PSD above 0.5 Hz ≥ 5.

The ratio threshold is below 0.5 because a Hann-windowed oscillation exactly at
3 or 12 Hz places about half its power outside the band. This avoids unstable
band-edge decisions while retaining amplitude and peak/background checks. These
are engineering defaults, not clinically calibrated detection thresholds.
Low signals return null frequency with measured power and zero rule fraction;
clean stillness is valid and does not force a retest.

## Limits

Broadband motion and intentional rhythmic movement can share these frequencies.
The rule does not identify the cause of an oscillation. Amplitudes depend on
placement, mounting and calibration. Fixed windows blur episodes, can miss weaker
simultaneous oscillations, and may select harmonics of non-sinusoidal motion.
Band-edge power and low signals are especially uncertain. Conservative segment
exclusion can request retest even when overall missingness is under 5%.
Real-data accuracy, confidence calibration, repeatability and minimum detectable
change remain unvalidated. Video, sway and repetition features remain unimplemented.

## Gait processing and event semantics

`extract_gait(recording, config)` applies gait quality gates and the same complete
segment preprocessing, using a 0.5–5 Hz band-pass. Metadata must identify `ankle`
or `wrist`. Metadata placement wins over `features.gait.placement`; the latter is
a fallback only when metadata lacks placement. Default canonical metadata says
wrist, so callers must explicitly mark ankle recordings.

Project band-passed gyroscope axes onto the highest-variance direction, computed
from the 3×3 covariance matrix over prepared windows. The sign is made
deterministic by making its largest loading positive. A signed known axis may
instead be configured. This handles fixed sensor rotations, but without mounting
calibration cannot establish anatomical mid-swing phase. It detects one positive
cycle peak rather than rectifying the signal and counting both lobes.

Default accepted windows require:

- projected gyro RMS ≥2 °/s;
- at least two positive peaks, separated by at least 0.4 s;
- peak prominence ≥max(5 °/s, 20% of window peak-to-peak angular velocity);
- all detected cycle intervals between 0.4 and 2 s;
- mean-centered lag correlation ≥0.3, using the median detected interval as lag.

These are provisional engineering thresholds, not a validated walking detector.
They can reject unusual or very irregular movement. At least three accepted
windows and 50% of expected windows are required; otherwise return `retest`.
No clinical status or task adherence is inferred from a periodic signal.

Peak times from overlapping windows are merged within 0.05 s (configurable).
Complete cycles count only consecutive deduplicated events whose pair was
observed together in at least one accepted window. No cycle is created across
excluded gaps, and overlapping window counts are never summed. Counts cover
accepted intervals only and are not extrapolated to the full task duration.

For ankle placement, one instrumented-limb cycle is treated as one stride,
corresponding to two inferred steps. For wrist placement, this relationship is
withheld by default: only swing cycles are reported. Set
`wrist_one_cycle_per_stride=true` only when that assumption is justified by the
guided protocol; it is not inferred from data.

## Gait metric definitions

| Name | Calculation / plain-language meaning | Unit / range |
| --- | --- | --- |
| `cadence_steps_min` | 120 / mean supported stride interval; inferred step frequency, ankle or explicitly mapped wrist only | steps/min; >0 or null |
| `mean_stride_time_s` | Mean time between consecutive supported same-limb events | s; >0 or null |
| `stride_time_cv` | Sample SD / mean of unique supported stride intervals; relative timing variability | Fraction; ≥0 or null |
| `number_of_strides` | Number of unique complete supported stride intervals, not number of peaks | Integer; ≥1 or null |
| `walking_speed_proxy_m_s` | Configured constant `assumed_stride_length_m` / mean stride time | m/s; >0 or null |
| `arm_swing_amplitude_deg` | Median cycle peak-to-peak integrated projected gyro angle in accepted wrist windows | degrees; ≥0 or null |
| `arm_swing_amplitude_cv` | Median window sample SD / mean of per-cycle angle amplitudes | Fraction; ≥0 or null |
| `arm_swing_cycles_min` | 60 / mean unique supported wrist cycle interval | cycles/min; >0 or null |
| `acceleration_rms_m_s2` | Median window vector RMS after gait-band filtering | m/s²; ≥0 |
| `turn_count` | Unavailable; no validated turn/orientation method implemented | null |

The angle proxy is computed using trapezoidal integration between successive
peaks, then subtracting a linear drift term that closes the integrated cycle.
It is not anatomical shoulder range of motion. No true left/right asymmetry can
be measured with this single sensor. Speed is null by default: without distance
calibration, timing and one IMU do not determine walking speed. A configured
constant stride length produces an explicit assumption-based proxy, not a
measured distance or speed. Changes in stride length are not captured.

Global cadence, mean stride time, stride CV and count use unique supported
intervals. Arm-swing cycle rate uses those same unique intervals; other amplitude
features use median window estimates. `spread` describes IQRs of **window
estimates**, not confidence intervals for global values. Per-window count IQR
does not quantify uncertainty in the global count. A window with fewer than two
intervals has null window CV even when CV can be computed over the global set.
Each spread entry includes its window-value count and recording quality score.

Event times, cycle intervals, projection direction, and each window's lag
correlation are retained as internal evidence. Reasons and limitations are
included in the internal result. This does not modify the proposed mobile API
schema. Features with insufficient evidence remain null. No typical clinical
ranges or clinical precision have been established for these estimates.

## IMU tapping processing

`extract_tapping_imu` requires wrist metadata and all six canonical channels.
The quality gate requires evidence of movement. Filter complete segments at
0.5–10 Hz using the shared preprocessing and 4-second, 50%-overlapping windows.
Default signal is acceleration; gyro can be selected explicitly. Auto projection
uses the highest-variance direction and selects its sign using the third central
moment so sparse pulses point upward. Alternatively configure a signed axis.
This is a pulse-motion convention, not an anatomical orientation estimate.

Default detection thresholds under `features.tapping_imu`:

- projected RMS ≥0.015 m/s² (accel) or 0.15 °/s (gyro);
- peak prominence ≥max(0.05 m/s² or 0.5 °/s, 10% of window peak-to-peak signal);
- peak separation ≥0.15 s;
- at least three peaks in a supported window;
- at least three supported windows and 50% of expected windows.

These engineering defaults are not clinically calibrated. No regularity gate is
applied: irregular rhythm and pauses are measurements of interest. Counts merge
overlapping-window detections within 0.04 s, independently per uninterrupted
prepared segment. Event amplitude is the median prominence of that event across
its window detections. Sparse windows can contribute events even if they cannot
provide a supported rhythm estimate.

Global intervals join successive unique events only within a common segment.
This lets a long pause between tapping runs count once while excluding missing
data. An interval can also represent a missed low-amplitude tap, so a hesitation
count cannot establish the cause. Preparation excludes unobserved margins and
partial windows; no event count is extrapolated to the full recording.

## IMU tapping metric definitions

Amplitude unit is m/s² by default, or °/s for gyro. These are filtered wrist-motion
proxies, not direct finger-opening distance or contact measurements.

| Name | Calculation / plain-language meaning | Unit / range |
| --- | --- | --- |
| `tap_count` | Count of unique detected events across prepared segments | Integer; ≥0 |
| `tap_rate_hz` | 1 / mean within-segment inter-event interval, including supported pauses | taps/s; >0 |
| `inter_tap_interval_cv` | Sample SD / mean of within-segment intervals | Fraction; ≥0 |
| `median_tap_amplitude` | Median event peak prominence | Selected sensor amplitude; >0 |
| `amplitude_slope_per_s` | Least-squares event amplitude slope versus elapsed seconds | Selected amplitude/s; signed or null |
| `amplitude_decrement_fraction` | 1 − fitted last amplitude / fitted first amplitude, over observed events | Dimensionless; signed or null |
| `hesitation_count` | Number of unique intervals strictly greater than 2× the median interval (configurable) | Integer; ≥0 |

Negative amplitude slope means declining motion amplitude. A negative decrement
fraction means increasing amplitude; values are not clipped into an assumed
decline. Linear fits describe the observed event span after filtering/trimming,
not the whole requested task. Global slope and decrement are null if events span
multiple discontinuous segments, or if the fit cannot be defined; within-window
trends remain available. No taps or insufficient supported windows returns
`retest`, rather than zero-rate or confident absent-symptom claims.

Output follows the internal feature-stage pattern with per-window evidence,
IQRs and quality score. Event times, amplitudes, segment IDs and included
intervals are retained for auditing. Window count/hesitation IQRs are not
uncertainty intervals for global event counts. Final confidence levels and
bootstrap intervals await the dedicated uncertainty stage.

Small impulses are smoothed by the band-pass: absolute amplitudes depend on tap
rate, sampling, mounting and pulse shape. The decrement tests use identical pulse
shapes with known changes. Real-data agreement with video, manual counts,
finger excursion and clinically relevant changes remains unvalidated.
