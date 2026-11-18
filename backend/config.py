"""Server settings. Override with environment variables, no code edits needed."""
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

# Trained model used at prediction time.
MODEL_PATH = Path(os.getenv("PATHSNAP_MODEL", ROOT / "models" / "artifacts" / "boosting_tuned.joblib"))

# Folder holding the trips the UI can pick from (searched recursively for *.track).
# TODO: point at the real folder. For dev use the student-test trips; on demo day
# use the evaluator trips.
TRACK_DIR = Path(os.getenv("PATHSNAP_TRACK_DIR", ROOT / "data" / "raw" / "student_test"))

# Second argument of src.data_io.load_trip(...): "student_test" in dev, "evaluator_test" on demo day.
PARTITION = os.getenv("PATHSNAP_PARTITION", "student_test")

# Set PATHSNAP_USE_HMM=0 to serve classifier-only predictions (until D's decoder is merged).
USE_HMM = os.getenv("PATHSNAP_USE_HMM", "1") == "1"
