# Roadmap and remaining work

Updated 2026-11-10. This is a student-project analysis pipeline with synthetic
engineering verification. It is not yet an app-integrated or clinically validated
system.

## Completed foundation

- Draft data contract and one configurable hardware/task settings file.
- CSV/JSON IMU loading, explicit packet-time reconstruction, phyphox mappings,
  unit/axis conversion and original timestamp audit.
- Seeded noise, tremor, tapping and walking fixtures plus artifacts.
- IMU quality score/reasons and task-aware retest gates.
- Anti-aliased resampling, gravity/offset filtering, independent complete
  segments and overlapping windows.
- Tremor features, single-sensor gait features and wrist IMU tapping features,
  with window IQRs, evidence and documented limitations.
- Tests for known signals, bad inputs and missing-signal abstention. Tests run
  on Python 3.13; Python 3.11 syntax is checked, execution remains unverified.

These are internal feature functions. The proposed mobile API, final confidence
levels, patient summaries and longitudinal analysis are not implemented yet.

## Implementation sequence

| Order | Remaining work | Completion evidence |
| --- | --- | --- |
| 1 | Video input, MediaPipe Hands/Pose extraction, video quality, finger tapping and video gait | Landmark-derived metrics tested with known trajectories; missing/occluded frames return retest; decode/extraction benchmark on representative phone clips |
| 2 | Postural sway | Configurable mounting/horizontal plane; RMS, path proxy, frequency and feasible ellipse metric; no claim of displacement without validated reconstruction |
| 3 | Exercise repetitions | Unique repetition counts, cycle durations/CV, motion-range proxy and slowing; synthetic count/rate/change recovery |
| 4 | Shared uncertainty | Consistent per-metric spread/count/quality, confidence levels with reasons, and justified bootstrap/95% intervals that account for dependent windows |
| 5 | Task API / strict JSON | One entry point per task; agreed app envelope, units/explanations, finite-or-null serialization, schema tests, retest with no metrics; optional server wrapper after deployment decision |
| 6 | Personal longitudinal baselines | First N valid-session median/MAD, robust change scores, consecutive-session rules; SEM/MDC enabled only with test–retest evidence |
| 7 | Template summaries | Patient and clinician text tied to computed metric names, neutral discussion prompts, no diagnostic statements; optional future rephrasing hook restricted to supplied values |
| 8 | Dataset adapters and small models | Verify PADS/mPower license/access/rates/placement/labels, write dataset cards, then add adapters; labels determine whether a non-diagnostic model is appropriate |
| 9 | Evaluation and validation | Subject-grouped splits, train-only normalization, versioned feature/model artifacts; accuracy/balanced accuracy/sensitivity/specificity/AUC with defensible CIs; leakage audit, ICC/SEM, MDC and failure analysis |
| 10 | Integration and reproducibility | Real wearable/phone contract tests, complete Python 3.11 environment, end-to-end CPU/memory benchmarks and documented demo recordings/results |

Each module is built and tested before the next, with results reported. Some
data gathering and contract decisions should happen alongside implementation;
real-data validation cannot be completed from synthetic fixtures alone.

## Team inputs still needed

**Wearable team:** actual IMU model/ranges/rate; raw units; axis/mounting
convention; timestamp clock/epoch/wrap/reset behavior; whether packet timestamps
describe sensor acquisition or phone reception; first/last sample anchor; actual
CSV/JSON examples and wrist/ankle/side metadata.

**App team:** final guided tests and instructions, identifiers/session boundaries,
video capture/rotation/frame timestamps, camera constraints, upload encoding,
phone-versus-server execution and the final response envelope. Keep current
defaults until these are confirmed. Proposed output-schema additions need review
before implementation; the feature-stage dictionaries are internal evidence.
No new dependency beyond the requested stack should be added without approval.

**Validation data:** manual tap/step/repetition counts, paired IMU/video where
possible, repeated recordings of the same participant/task, and subject IDs for
evaluation. Never put windows from one participant in both train and test.
Dataset access and permissible use must be verified before use or redistribution.

## Known limits to carry forward

- Rule thresholds are provisional; passing synthetic tests does not establish
  real-world accuracy or clinically meaningful changes.
- Wrist peaks are tap/swing proxies, not direct finger kinematics. Ankle cycles
  assume one event per stride. One sensor cannot measure bilateral asymmetry.
- Gait speed is withheld without stride-length assumptions; turns are unavailable.
- Current IQRs describe dependent windows, not calibrated confidence intervals.
- App schema, video runtime and the under-one-second complete recording target
  remain unverified. Local IMU feature timing excludes I/O and import startup.
- No diagnosis, disease probability or MDS-UPDRS score belongs in any stage.
