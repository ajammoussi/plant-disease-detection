"""
data_cleaning.py
================
Detect and remove corrupt, duplicate, or low-quality images from PlantVillage.
"""

import os
import hashlib
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np
from PIL import Image, UnidentifiedImageError


# ── 1. Corrupt image detection ───────────────────────────────────────────────
def check_image_integrity(path: Path) -> Tuple[bool, Optional[str]]:
    """
    Try to fully load and verify a single image.

    Returns
    -------
    (is_valid, reason)
        reason is None if valid, otherwise a short description of the problem.
    """
    try:
        with Image.open(path) as img:
            img.verify()           # catches truncated / corrupt headers
        # Re-open after verify (verify() closes/invalidates the handle)
        with Image.open(path) as img:
            img.load()             # forces full pixel decoding
        return True, None
    except (UnidentifiedImageError, OSError) as e:
        return False, f"corrupt: {e}"
    except Exception as e:
        return False, f"unexpected error: {e}"


def find_corrupt_images(
    class_map: Dict[str, List[Path]],
    verbose: bool = False,
) -> Dict[str, List[Tuple[Path, str]]]:
    """
    Scan every image and report corrupt ones.

    Parameters
    ----------
    class_map : dict[class_name, list[Path]]
    verbose : bool
        Print progress per class.

    Returns
    -------
    dict[class_name, list[(path, reason)]]
        Only classes with at least one bad image are included.
    """
    report: Dict[str, List[Tuple[Path, str]]] = {}
    for cls, paths in class_map.items():
        bad = []
        for p in paths:
            ok, reason = check_image_integrity(p)
            if not ok:
                bad.append((p, reason))
        if bad:
            report[cls] = bad
            if verbose:
                print(f"  [{cls}] {len(bad)} corrupt image(s)")
    return report


def remove_corrupt_images(
    corrupt_report: Dict[str, List[Tuple[Path, str]]],
    class_map: Dict[str, List[Path]],
    dry_run: bool = True,
) -> Dict[str, List[Path]]:
    """
    Remove corrupt images from the class_map (and optionally from disk).

    Parameters
    ----------
    corrupt_report : output of find_corrupt_images
    class_map : original class_map (will NOT be mutated)
    dry_run : if True only report; if False delete files from disk.

    Returns
    -------
    Cleaned class_map.
    """
    bad_set = {p for paths in corrupt_report.values() for p, _ in paths}
    cleaned: Dict[str, List[Path]] = {}
    for cls, paths in class_map.items():
        good = [p for p in paths if p not in bad_set]
        cleaned[cls] = good
        if not dry_run:
            for p in paths:
                if p in bad_set and p.exists():
                    p.unlink()

    n_removed = sum(len(v) for v in corrupt_report.values())
    action = "Would remove" if dry_run else "Removed"
    print(f"[cleaning] {action} {n_removed} corrupt image(s).")
    return cleaned


# ── 2. Duplicate detection (perceptual hash) ─────────────────────────────────
def _md5_hash(path: Path, chunk_size: int = 8192) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(chunk_size), b""):
            h.update(chunk)
    return h.hexdigest()


def find_exact_duplicates(
    class_map: Dict[str, List[Path]],
) -> List[List[Path]]:
    """
    Find groups of byte-identical images using MD5 hashing.

    Returns
    -------
    list of groups, each group being a list of duplicate paths (≥2).
    """
    hash_map: Dict[str, List[Path]] = {}
    for paths in class_map.values():
        for p in paths:
            try:
                h = _md5_hash(p)
                hash_map.setdefault(h, []).append(p)
            except OSError:
                pass
    return [group for group in hash_map.values() if len(group) > 1]


def remove_duplicates(
    class_map: Dict[str, List[Path]],
    duplicate_groups: List[List[Path]],
    dry_run: bool = True,
) -> Dict[str, List[Path]]:
    """
    Keep the first file in each duplicate group, mark the rest for removal.

    Parameters
    ----------
    dry_run : if False, delete the duplicate files from disk.

    Returns
    -------
    Cleaned class_map.
    """
    to_remove = {p for group in duplicate_groups for p in group[1:]}
    cleaned: Dict[str, List[Path]] = {}
    for cls, paths in class_map.items():
        cleaned[cls] = [p for p in paths if p not in to_remove]
        if not dry_run:
            for p in paths:
                if p in to_remove and p.exists():
                    p.unlink()

    action = "Would remove" if dry_run else "Removed"
    print(f"[cleaning] {action} {len(to_remove)} duplicate image(s).")
    return cleaned


# ── 3. Low-quality / too-small image filter ──────────────────────────────────
def find_low_quality_images(
    class_map: Dict[str, List[Path]],
    min_size: Tuple[int, int] = (32, 32),
    min_file_bytes: int = 1_000,
) -> Dict[str, List[Tuple[Path, str]]]:
    """
    Flag images that are too small (resolution or file size).

    Parameters
    ----------
    min_size : (width, height) minimum acceptable resolution.
    min_file_bytes : minimum acceptable file size in bytes.

    Returns
    -------
    dict[class_name, list[(path, reason)]]
    """
    report: Dict[str, List[Tuple[Path, str]]] = {}
    for cls, paths in class_map.items():
        bad = []
        for p in paths:
            if p.stat().st_size < min_file_bytes:
                bad.append((p, f"file too small ({p.stat().st_size} B)"))
                continue
            try:
                with Image.open(p) as img:
                    w, h = img.size
                if w < min_size[0] or h < min_size[1]:
                    bad.append((p, f"resolution too low ({w}×{h})"))
            except Exception as e:
                bad.append((p, str(e)))
        if bad:
            report[cls] = bad
    return report


# ── 4. Summary report ────────────────────────────────────────────────────────
def cleaning_summary(
    original_class_map: Dict[str, List[Path]],
    cleaned_class_map: Dict[str, List[Path]],
) -> Dict:
    """Return a dict summarising what was removed."""
    orig_total = sum(len(v) for v in original_class_map.values())
    clean_total = sum(len(v) for v in cleaned_class_map.values())
    return {
        "original_total": orig_total,
        "cleaned_total": clean_total,
        "removed": orig_total - clean_total,
        "removal_pct": 100 * (orig_total - clean_total) / orig_total if orig_total else 0,
    }
