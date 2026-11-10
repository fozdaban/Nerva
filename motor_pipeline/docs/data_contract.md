# Proposed data contract — version 0.1.0

This is the first-stage proposal for the wearable, app and analysis boundary.
The input adapter is implemented. The task output below preserves the requested
schema and is a target for later modules, not an implemented analysis API.
Outputs describe motor measurements, never diagnoses, disease probabilities or
MDS-UPDRS scores. Clinical interpretations and thresholds are outside this stage.

## Input: canonical IMU recording

`Recording.data` is a pandas DataFrame in acquisition time order after configured
timestamp repair, with exactly these float columns:

| Column | Meaning | Canonical unit |
| --- | --- | --- |
| t | Elapsed sample time, starting at 0 | s |
| ax, ay, az | Acceleration, including gravity if present in source | m/s² |
| gx, gy, gz | Angular velocity | °/s |

Sensor values may be NaN (missing); timestamps must be finite. Infinity in sensor
values becomes NaN and is counted. No filtering, resampling, gravity removal or
missing-value imputation happens at ingestion. Timestamp gaps remain gaps.
Source channels must share a synchronized time base. Separately sampled channels
must be aligned explicitly in a future source adapter, never joined by row number.

`Recording.metadata` includes `device`, `placement`, `side`, `sampling_rate_hz`,
`task_name`, `session_id`, `user_id`, `observed_sampling_rate_hz`,
`acceleration_unit` and `gyroscope_unit`. File loaders also add `source_format`
and `input_profile`. Canonical units and configured sample rate are adapter-owned.
Other metadata may be supplied by a JSON envelope or the caller; caller values
override the envelope. IDs are opaque strings; their defaults are null for local
fixtures. Require nonempty user/session IDs before longitudinal analysis or model
evaluation. No identifying details are needed in IDs.

Placement examples: wrist, ankle, lower_back, chest. Side examples: left, right,
unknown. No orientation is inferred. Sway's horizontal axes need a known mounting
or a later gravity alignment step. Metadata defaults are placeholders, not evidence
that the module was worn in that position.

## External formats and settings

`motor_pipeline/config.yaml` is JSON-compatible YAML, read with the standard
library. Keep JSON syntax when editing; no PyYAML dependency is needed. `input`
holds hardware, units, timestamp policy and file mappings. `metadata_defaults`
holds provisional placement/side/task. Rate defaults to 100 Hz and supports
50–200 Hz. Full-scale defaults are ±8 g and ±1000 °/s. These are configurable
assumptions; the actual hardware range is not inferred from samples.

Standard CSV has a header `t,ax,ay,az,gx,gy,gz`. Map arbitrary export headers
through `input.columns`, whose keys are canonical names and values are source
names. Configure `file_format` (`auto`, `csv`, `json`) and `csv_delimiter`.
Numeric timestamp units: `s`, `ms`, `us`, `ns`. `iso8601` timestamps require
explicit timezone offsets; numeric times may be relative or Unix epoch times.
Internally times use float64 seconds; epoch nanoseconds lose sub-microsecond
precision during conversion. That resolution is well below the 5–20 ms sample
period, but this is not a nanosecond-preserving archival format.

Acceleration units: `m/s2` or `g`; angular velocity: `deg/s` or `rad/s`.
`axis_map` is a signed permutation within each sensor, e.g. `"ax": "-ay"`;
both sensor maps must remain permutations. Arbitrary calibration matrices are
not implemented. Confirm units from the exporter; the adapter does not guess.

JSON sample input may be a list of sample objects or this envelope:

```json
{
  "metadata": {"device": "prototype", "placement": "wrist", "side": "right", "task_name": "rest_tremor", "session_id": "s001", "user_id": "u001"},
  "samples": [
    {"t": 0.00, "ax": 0.0, "ay": 0.0, "az": 9.80665, "gx": 0.1, "gy": 0.0, "gz": 0.0},
    {"t": 0.01, "ax": 0.1, "ay": 0.0, "az": 9.80665, "gx": 0.2, "gy": 0.0, "gz": 0.0}
  ]
}
```

JSON packet input uses explicit packet boundaries:

```json
{"packets": [{"timestamp": 10.0, "samples": [
  {"ax": 0, "ay": 0, "az": 9.8, "gx": 0, "gy": 0, "gz": 0},
  {"ax": 0, "ay": 0, "az": 9.8, "gx": 0, "gy": 0, "gz": 0}
]}]}
```

The packet timestamp denotes the first sample by default; `packet_timestamp_anchor`
may be `last`. Reconstruct within-packet times at `1 / sampling_rate_hz`, anchored
to that packet, without stretching across lost packets. Actual within-packet
jitter cannot be recovered. This assumes the sensor clock rate is correct;
phone reception timestamps can introduce jitter and should be identified by the
firmware team. Per-sample time fields in packet samples are replaced by the
reconstructed times. At 100 Hz the example becomes t=[0, 0.01].

For a CSV with repeated packet timestamps, set `timestamp_mode` to `packet`;
consecutive equal times form a packet. Adjacent packets with identical timestamps
cannot be distinguished in that CSV representation; prefer JSON packets or
per-sample timestamps. JSON `packets` always enables packet expansion.

## Audit and repair

`Recording.diagnostics` records original sample count, retained sample count,
duplicate reconstructed times, backward transitions, whether repair was applied,
packet count where applicable, original time origin, gaps, estimated missing
samples and time fraction, nonfinite sensor count/fraction, and absent channels.

Default `timestamp_repair=sort_deduplicate` stably sorts timestamps, keeping the
first acquired row for each duplicate time. `reject` raises `InputError` for
duplicates or backward time. This repairs packet reordering, not sensor-clock
resets; clocks that restart or wrap need a source-specific adapter and should
use rejection. A gap is strictly more than `quality.gap_periods` nominal periods
(default 2, with a numerical tolerance). Estimated lost samples within those
gaps are sum(max(gap_seconds × configured_rate − 1, 0)); missing time fraction
divides this estimate by retained plus estimated lost samples. Subthreshold
jitter/short omissions are not counted by this estimate. Missing-channel and
nonfinite-value fractions are separate evidence, not a quality score.

Malformed input raises `InputError`; file/JSON/CSV parsing failures may raise
their native exceptions. Empty input, missing timestamps and absent required
channels cannot safely be converted. A future task API should translate these
into retest/invalid-input responses without inventing metrics.

## Phone / phyphox

`load_imu(path, profile="phyphox")` applies `input.phyphox` mappings. Defaults
target a **combined synchronized CSV** with `Time (s)`, `Acceleration x (m/s^2)`
(and y/z) and `Gyroscope x (rad/s)` (and y/z). Actual phyphox exports differ by
experiment, platform and selected sensor. Update mappings from an actual export;
the defaults do not claim universal phyphox format support. Acceleration with
gravity and linear acceleration are not interchangeable: record the experiment
and whether gravity was already removed in metadata.

For acceleration-only exports, explicitly set `allow_missing_gyroscope=true`.
Missing gyroscope channels become NaN and are listed in diagnostics. Later tasks
must reject missing signals they require; absent channels are never zero-filled.
Separate multi-file and ZIP exports are not supported in this first stage.
Public dataset adapters will be added after license/access/source verification.

## Proposed task output (later stage)

Keep the requested top-level fields below. `api.field_names` holds configurable
top-level aliases pending app-team confirmation. The current ingestion module
does not serialize task results. The following numbers are illustrative only:

```json
{
  "task": "rest_tremor",
  "session_id": "s001",
  "status": "ok",
  "quality": {"score": 0.92, "reasons": []},
  "metrics": {
    "dominant_freq_hz": {
      "value": 5.1,
      "ci95": [4.8, 5.4],
      "explanation": "Frequency of the strongest peak in the configured tremor band."
    }
  },
  "confidence": "high",
  "change_from_baseline": null,
  "summary_patient": "The strongest oscillation was 5.1 Hz (dominant_freq_hz).",
  "summary_clinician": "Dominant tremor-band peak: 5.1 Hz (dominant_freq_hz).",
  "pipeline_version": "0.1.0"
}
```

JSON must use finite numbers or null, never NaN/Infinity. Poor quality must return
`status: "retest"` with reasons and no numerical metrics (`metrics: {}` in a
full envelope). Never calculate a confident metric from missing signals.
Retain `quality.score` on every full result. The requested per-metric window
counts, spread, units, per-window values and confidence reasons are still to be
reconciled with the example metric schema; additions need your approval before
implementation. IQR must not be labeled `ci95`. Video payload/upload encoding
and server execution remain TBD. No FastAPI or video dependency is introduced
in this stage.
