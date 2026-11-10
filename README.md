# Motor pipeline

Student-project foundation for interpretable wearable motor measurements.
Implemented stages: proposed data contract/config, IMU input adapter, seeded
synthetic fixtures, task-aware IMU quality gates, preprocessing and windowed
tremor and gait features. Tapping, sway, repetitions and video quality checks
are **not implemented yet**.
No diagnoses, disease probabilities or clinical scores are produced.

Use Python 3.11 in a virtual environment:

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\python -m pip install -e ".[test]"
.\.venv\Scripts\python -m pytest
```

NumPy, pandas and SciPy are needed at runtime; pytest is used for
tests. Scikit-learn, MediaPipe and OpenCV belong to later stages. No other
heavy dependency or YAML parser is added. Python >=3.11 is accepted by packaging;
3.11 remains the target. See validation notes for the interpreter actually tested.

```python
from motor_pipeline.config import load_config
from motor_pipeline.io import load_imu, normalize_imu
from motor_pipeline.synthetic import tremor, add_artifact

config = load_config()
frame = tremor(frequency_hz=5.0, sampling_rate_hz=100, seed=42)
frame = add_artifact(frame, "gap", start_s=5, duration_s=0.2)
recording = normalize_imu(frame, config, metadata={
    "user_id": "u001", "session_id": "s001", "task_name": "rest_tremor"
})
print(recording.diagnostics)
from motor_pipeline.quality import assess_imu

quality = assess_imu(recording, config=config)
print(quality.as_dict())  # {"score": ..., "reasons": [...]}
if quality.status == "retest":
    print(quality.retest_response())  # No motor metrics on rejected recordings.
frame.to_csv("example.csv", index=False)
loaded = load_imu("example.csv", config)
# For a matching phone export:
# loaded = load_imu("phone.csv", config, profile="phyphox")
```

Read [the proposed contract](motor_pipeline/docs/data_contract.md) before sharing
recordings and edit [config.yaml](motor_pipeline/config.yaml) for hardware/export
changes. The file uses JSON syntax (valid YAML) to avoid adding dependencies.
Metadata placeholders do not establish actual placement or task identity.

Synthetic generators: `resting_noise`, `tremor`, `tapping`, `walking` return
canonical DataFrames. `add_artifact` injects gaps, clipping, sudden movement,
missing values, duplicate samples or reversed sample order. Seeded noise is
reproducible. These are algorithm fixtures, not realistic disease simulations.

Read [the quality checks](motor_pipeline/docs/quality.md) for the score formula,
task requirements and configurable engineering thresholds.

Extract tremor features from a canonical recording:

```python
from motor_pipeline.features import extract_tremor

result = extract_tremor(recording, config)
if result["status"] == "ok":
    print(result["features"]["dominant_freq_hz"])
    print(result["spread"]["dominant_freq_hz"])
else:
    print(result["reasons"])
```

This returns internal named features and per-window evidence, not the future
mobile API envelope. Low-signal frequencies are null; low-quality recordings
return `retest` without features. See [metric definitions](motor_pipeline/docs/metrics.md)
for units, explanations and the provisional oscillation rule. IQRs describe
window spread, not 95% confidence intervals. No clinical validity is established.

Extract gait features with explicit sensor placement:

```python
from motor_pipeline.synthetic import walking
from motor_pipeline.features import extract_gait

ankle = normalize_imu(walking(), config, metadata={"placement": "ankle"})
gait = extract_gait(ankle, config)
if gait["status"] == "ok":
    print(gait["features"]["cadence_steps_min"])
    print(gait["features"]["number_of_strides"])
    print(gait["limitations"])
```

The ankle method maps one positive cycle to one stride (two steps). Counts
deduplicate overlapping windows and cover only complete observed cycles.
Wrist mode reports swing timing and integrated angle proxies; stride metrics
require the explicit `wrist_one_cycle_per_stride` assumption. Speed remains null
without a configured stride length; turn counting remains unavailable.
Canonical metadata defaults to wrist, so ankle inputs must set placement.

Next stages, each with tests and a progress report:
IMU/video tapping → sway → repetitions → uncertainty → task API → personal
baselines → summaries → dataset/model evaluation → validation. Dataset licenses,
clinical performance and phone processing speeds have not yet been assessed.
