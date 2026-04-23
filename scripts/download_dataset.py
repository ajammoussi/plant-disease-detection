
from pathlib import Path
import numpy as np
import sys
import os
import random
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib

PROJECT_ROOT = Path.cwd().parent  # scripts/ → project root
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src import data_loader, eda, data_cleaning, preprocessing

# ── Paths configuration ────────────────────────────────────────────────────
DATA_DIR    = PROJECT_ROOT / "data"                         # where to store raw data
FIGURES_DIR = PROJECT_ROOT / "outputs" / "figures"          # where to save figures

DATA_DIR.mkdir(parents=True, exist_ok=True)
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

# Seed for reproducibility
SEED = 42
random.seed(SEED)
np.random.seed(SEED)

print(f"Data    → {DATA_DIR}")
print(f"Figures → {FIGURES_DIR}")

# ── Download dataset ───────────────────────────────────────────────────────
# Source URL (Kaggle API — no login required for this public dataset)
KAGGLE_URL = "https://www.kaggle.com/api/v1/datasets/download/abdallahalidev/plantvillage-dataset"

# Only download if not already present
raw_dir = DATA_DIR / "raw"
if not raw_dir.exists() or not any(raw_dir.rglob("*.jpg")):
    print("Downloading PlantVillage dataset (~1.2 GB) …")
    extract_path = data_loader.download_dataset(dest_dir=str(DATA_DIR), url=KAGGLE_URL)
else:
    print("Dataset already present, skipping download.")
    extract_path = str(raw_dir)

# ── Locate dataset root (folder of class sub-directories) ─────────────────
dataset_root = data_loader.find_dataset_root(extract_path)
print(f"Dataset root found: {dataset_root}")
print(f"Sub-directories   : {[d.name for d in sorted(dataset_root.iterdir()) if d.is_dir()][:5]} …")
