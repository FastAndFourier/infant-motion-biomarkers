# Configure these paths to point to your local data directory.
# Set the CP_DATA_DIR environment variable, or edit YOUR_DATA_DIR below.
import os
YOUR_DATA_DIR = os.environ.get("CP_DATA_DIR", "/path/to/your/data")

DATASET_PATH          = f"{YOUR_DATA_DIR}"
RING_DATASET_PATH     = f"{YOUR_DATA_DIR}/ringDataset"
IMU_DATASET_PATH      = f"{YOUR_DATA_DIR}/imuDataset"
GRASP_ANNOT_DIR       = f"{YOUR_DATA_DIR}/grasping_annotation"
OUTCOME_PATH          = f"{YOUR_DATA_DIR}/clinical_outcome.csv"
OUTPUT_ALIGN_DIR      = f"{YOUR_DATA_DIR}/aligned"
OUTPUT_UNLABELED_DIR  = f"{YOUR_DATA_DIR}/unlabeled"
OUTPUT_PREDICTION_DIR = f"{YOUR_DATA_DIR}/prediction"
EXTERNAL_DATA_DIR     = f"{YOUR_DATA_DIR}/external"
VIDEO_TS_DIR          = f"{YOUR_DATA_DIR}/timestamps"
TOKENIZED_DIR         = f"{YOUR_DATA_DIR}/tokenized"
