"""
data_loader.py
==============
Handles dataset acquisition and raw loading for PlantVillage.
"""

import os
import zipfile
import shutil
import random
from pathlib import Path
from typing import Tuple, Dict, List, Optional

import numpy as np
from PIL import Image


# ── Constants ──────────────────────────────────────────────────────────────
DATASET_URL = "https://www.kaggle.com/api/v1/datasets/download/emmarex/plantdisease"
SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff"}


# ── Download ────────────────────────────────────────────────────────────────
def download_dataset(dest_dir: str, url: str = DATASET_URL) -> str:
    """
    Download the PlantVillage zip from Kaggle and extract it.

    Parameters
    ----------
    dest_dir : str
        Root folder where data/ will be created.
    url : str
        Kaggle dataset download URL.

    Returns
    -------
    str
        Path to the extracted dataset root directory.
    """
    import requests

    dest_dir = Path(dest_dir)
    dest_dir.mkdir(parents=True, exist_ok=True)
    zip_path = dest_dir / "plantdisease.zip"

    print(f"[download] Downloading dataset from:\n  {url}")
    with requests.get(url, stream=True) as r:
        r.raise_for_status()
        total = int(r.headers.get("content-length", 0))
        downloaded = 0
        with open(zip_path, "wb") as f:
            for chunk in r.iter_content(chunk_size=8192):
                f.write(chunk)
                downloaded += len(chunk)
                if total:
                    pct = 100 * downloaded / total
                    print(f"\r  {pct:.1f}%", end="", flush=True)
    print(f"\n[download] Saved to {zip_path}")

    extract_dir = dest_dir / "raw"
    print(f"[download] Extracting to {extract_dir} ...")
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(extract_dir)
    print("[download] Done.")
    return str(extract_dir)


# ── Dataset scanning ────────────────────────────────────────────────────────
def find_dataset_root(search_dir: str) -> Path:
    """
    Walk *search_dir* looking for the first directory that contains
    multiple sub-directories each holding image files (i.e. the class folders).
    """
    search_dir = Path(search_dir)
    for root, dirs, files in os.walk(search_dir):
        root_p = Path(root)
        subdirs_with_images = [
            d for d in dirs
            if any(
                f.suffix.lower() in SUPPORTED_EXTENSIONS
                for f in (root_p / d).iterdir()
                if f.is_file()
            )
        ]
        if len(subdirs_with_images) >= 5:
            return root_p
    # Fallback
    return search_dir

def split_segmented_originals(
    class_map: Dict[str, List[Path]],
    black_pixel_threshold: float = 0.08,
    black_value_cutoff: int = 20,
) -> Tuple[Dict[str, List[Path]], Dict[str, List[Path]]]:
    """
    Separate original images from pre-segmented (black-background) ones
    using whole-image black pixel ratio. Border-strip detection was found
    to over-filter classes with natural dark edges (corn, peach).
    """
    originals: Dict[str, List[Path]] = {}
    segmented: Dict[str, List[Path]] = {}
    for cls, paths in class_map.items():
        orig, seg = [], []
        for p in paths:
            try:
                arr = np.array(Image.open(p).convert("RGB"))
                black_ratio = np.mean(np.all(arr < black_value_cutoff, axis=2))
                (seg if black_ratio > black_pixel_threshold else orig).append(p)
            except Exception:
                orig.append(p)
        originals[cls] = orig
        segmented[cls] = seg
    return originals, segmented

def scan_dataset(dataset_root: str) -> Dict[str, List[Path]]:
    """
    Scan the dataset directory and return a dict mapping class_name → list of image paths.

    Parameters
    ----------
    dataset_root : str
        Path to the folder whose immediate sub-directories are class folders.

    Returns
    -------
    dict[str, list[Path]]
    """

    root = Path(dataset_root)
    class_map: Dict[str, List[Path]] = {}

    for cls_dir in sorted(root.iterdir()):
        if not cls_dir.is_dir():
            continue
        # Skip meta-folders that are themselves dataset roots (nested structure)
        sub_subdirs = [d for d in cls_dir.iterdir() if d.is_dir()]
        direct_images = [
            p for p in cls_dir.iterdir()
            if p.is_file() and p.suffix.lower() in SUPPORTED_EXTENSIONS
        ]
        if sub_subdirs and not direct_images:
            continue
        images = [
            p for p in cls_dir.rglob("*")
            if p.is_file() and p.suffix.lower() in SUPPORTED_EXTENSIONS
        ]
        if images:
            class_map[cls_dir.name] = images

    return class_map


def get_dataset_stats(class_map: Dict[str, List[Path]]) -> Dict:
    """
    Compute high-level statistics from the class map.

    Returns
    -------
    dict with keys: total_images, num_classes, class_counts,
                    min_count, max_count, mean_count
    """
    counts = {cls: len(paths) for cls, paths in class_map.items()}
    total = sum(counts.values())
    values = list(counts.values())
    return {
        "total_images": total,
        "num_classes": len(counts),
        "class_counts": counts,
        "min_count": min(values),
        "max_count": max(values),
        "mean_count": float(np.mean(values)),
    }


# ── Sample loading ──────────────────────────────────────────────────────────
def load_sample_images(
    class_map: Dict[str, List[Path]],
    n_per_class: int = 3,
    seed: int = 42,
) -> Dict[str, List[np.ndarray]]:
    """
    Load *n_per_class* random images (as RGB numpy arrays) for each class.

    Parameters
    ----------
    class_map : dict
    n_per_class : int
    seed : int

    Returns
    -------
    dict[class_name, list[np.ndarray]]
    """
    rng = random.Random(seed)
    samples: Dict[str, List[np.ndarray]] = {}
    for cls, paths in class_map.items():
        chosen = rng.sample(paths, min(n_per_class, len(paths)))
        samples[cls] = [np.array(Image.open(p).convert("RGB")) for p in chosen]
    return samples


def load_image(path: str) -> np.ndarray:
    """Load a single image as an RGB numpy array."""
    return np.array(Image.open(path).convert("RGB"))
