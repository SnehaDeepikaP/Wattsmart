"""Configuration module for Wattsmart.

Provides environment-based configuration, path definitions, and modeling constants.
No hardcoded machine-specific paths or debug=True flags.
"""

import os
from pathlib import Path
from typing import List

from dotenv import load_dotenv

# Load optional .env file
load_dotenv()

# Base project directories
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_PATH = DATA_DIR / "raw" / "Energy_Prediction.csv"
PROCESSED_DATA_PATH = DATA_DIR / "processed" / "energy_cleaned.csv"
MODELS_DIR = PROJECT_ROOT / "models"
MODEL_PATH = MODELS_DIR / "model.joblib"
METRICS_PATH = MODELS_DIR / "metrics.json"
FEATURE_LIST_PATH = MODELS_DIR / "feature_list.json"
SHAP_SUMMARY_PATH = MODELS_DIR / "shap_summary.png"
FIGURES_DIR = PROJECT_ROOT / "docs" / "figures"

# Ensure runtime directories exist
DATA_DIR.mkdir(parents=True, exist_ok=True)
(DATA_DIR / "raw").mkdir(parents=True, exist_ok=True)
(DATA_DIR / "processed").mkdir(parents=True, exist_ok=True)
MODELS_DIR.mkdir(parents=True, exist_ok=True)
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

# Application settings
APP_ENV = os.getenv("APP_ENV", "production")
DEBUG = os.getenv("DEBUG", "false").lower() in ("true", "1", "yes")
HOST = os.getenv("HOST", "0.0.0.0")
PORT = int(os.getenv("PORT", "7860" if os.getenv("SPACE_ID") else "5000"))
SECRET_KEY = os.getenv("SECRET_KEY", "wattsmart-production-secret-key-42")
CORS_ORIGINS = [
    origin.strip()
    for origin in os.getenv("CORS_ORIGINS", "*").split(",")
    if origin.strip()
]

# ML Pipeline Constants
TARGET_COL = "Appliances"
RANDOM_STATE = 42
TRAIN_TEST_SPLIT_RATIO = 0.80
TS_CV_SPLITS = 5
RESAMPLE_FREQ = "10min"  # 10-minute mean aggregation verified by validation results
CONFORMAL_CONFIDENCE_LEVEL = 0.90  # 90% prediction interval coverage

# Columns to drop with clear rationale
REDUNDANT_COLS = ["T2", "RH_2"]  # T2 == T_out and RH_2 == RH_out exactly
NOISE_COLS = ["rv1", "rv2"]  # Synthetic random uniform noise variables from UCI dataset

# Indoor sensor groupings
INDOOR_TEMP_COLS: List[str] = ["T1", "T3", "T4", "T5", "T7", "T8", "T9"]
INDOOR_RH_COLS: List[str] = ["RH_1", "RH_3", "RH_4", "RH_5", "RH_7", "RH_8", "RH_9"]
OUTDOOR_WEATHER_COLS: List[str] = [
    "T_out",
    "Press_mm_hg",
    "RH_out",
    "Windspeed",
    "Visibility",
    "Tdewpoint",
]
