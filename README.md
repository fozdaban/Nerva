# Motor pipeline

Student-project foundation for interpretable wearable motor measurements.
Implemented stage: proposed data contract/config, IMU input adapter and seeded
synthetic fixtures. Quality scoring and motor analysis are **not implemented yet**.
No diagnoses, disease probabilities or clinical scores are produced.

Use Python 3.11 in a virtual environment:

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\python -m pip install -e ".[test]"
.\.venv\Scripts\python -m pytest
```

Only NumPy and pandas are needed at runtime in this stage; pytest is used for
tests. SciPy, scikit-learn, MediaPipe and OpenCV belong to later stages. No other
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

Next stages, each with tests and a progress report: quality → tremor → gait →
IMU/video tapping → sway → repetitions → uncertainty → task API → personal
baselines → summaries → dataset/model evaluation → validation. Dataset licenses,
clinical performance and phone processing speeds have not yet been assessed.
