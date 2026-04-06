"""
eda.py
======
Exploratory Data Analysis utilities for PlantVillage.
Produces matplotlib figures — no display side-effects; callers call plt.show().
"""

from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from PIL import Image
import random


# ── Palette helper ──────────────────────────────────────────────────────────
def _class_palette(n: int):
    cmap = plt.colormaps["tab20"].resampled(n)
    return [cmap(i) for i in range(n)]


# ── 1. Class distribution ────────────────────────────────────────────────────
def plot_class_distribution(
    class_counts: Dict[str, int],
    figsize: Tuple[int, int] = (18, 7),
    title: str = "PlantVillage — Class Distribution",
    save_path: Optional[str] = None,
) -> plt.Figure:
    """
    Horizontal bar chart of image counts per class.
    """
    classes = list(class_counts.keys())
    counts = list(class_counts.values())
    order = np.argsort(counts)[::-1]

    classes = [classes[i] for i in order]
    counts = [counts[i] for i in order]
    colors = _class_palette(len(classes))

    fig, ax = plt.subplots(figsize=figsize)
    bars = ax.barh(classes, counts, color=colors, edgecolor="white", linewidth=0.4)
    ax.set_xlabel("Number of Images", fontsize=12)
    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.invert_yaxis()

    for bar, cnt in zip(bars, counts):
        ax.text(
            bar.get_width() + 10, bar.get_y() + bar.get_height() / 2,
            str(cnt), va="center", ha="left", fontsize=7,
        )

    ax.spines[["top", "right"]].set_visible(False)
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig


# ── 2. Species vs disease breakdown ─────────────────────────────────────────
def parse_class_names(class_names: List[str]) -> Tuple[Dict, Dict]:
    """
    Split PlantVillage class names (format: 'Species___Disease') into
    (species_counts, disease_counts) dicts.
    """
    species_counts: Dict[str, int] = {}
    disease_counts: Dict[str, int] = {}
    for cls in class_names:
        parts = cls.split("___")
        species = parts[0].replace("_", " ")
        disease = parts[1].replace("_", " ") if len(parts) > 1 else "Unknown"
        species_counts[species] = species_counts.get(species, 0) + 1
        disease_counts[disease] = disease_counts.get(disease, 0) + 1
    return species_counts, disease_counts


def plot_species_disease_breakdown(
    class_names: List[str],
    figsize: Tuple[int, int] = (16, 6),
    save_path: Optional[str] = None,
) -> plt.Figure:
    """Pie charts for species and disease distributions."""
    species_counts, disease_counts = parse_class_names(class_names)

    fig, axes = plt.subplots(1, 2, figsize=figsize)

    for ax, counts, title in zip(
        axes,
        [species_counts, disease_counts],
        ["Plant Species", "Disease / Condition"],
    ):
        labels = list(counts.keys())
        sizes = list(counts.values())
        colors = _class_palette(len(labels))
        wedges, texts, autotexts = ax.pie(
            sizes, labels=None, colors=colors,
            autopct=lambda p: f"{p:.1f}%" if p > 3 else "",
            startangle=140, pctdistance=0.80,
        )
        ax.legend(
            wedges, labels, loc="lower center",
            bbox_to_anchor=(0.5, -0.25), ncol=2, fontsize=7, frameon=False,
        )
        ax.set_title(title, fontsize=13, fontweight="bold")

    fig.suptitle("Taxonomy Breakdown", fontsize=15, fontweight="bold", y=1.02)
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig


# ── 3. Image grid sampler ────────────────────────────────────────────────────
def plot_sample_grid(
    class_map: Dict[str, List[Path]],
    classes_to_show: Optional[List[str]] = None,
    n_per_class: int = 4,
    seed: int = 42,
    figsize_per_img: float = 2.2,
    save_path: Optional[str] = None,
) -> plt.Figure:
    """
    Show a grid of sample images; rows = classes, cols = samples.
    """
    rng = random.Random(seed)
    if classes_to_show is None:
        classes_to_show = rng.sample(list(class_map.keys()), min(8, len(class_map)))

    n_rows = len(classes_to_show)
    n_cols = n_per_class
    fig, axes = plt.subplots(
        n_rows, n_cols,
        figsize=(figsize_per_img * n_cols, figsize_per_img * n_rows),
    )
    if n_rows == 1:
        axes = [axes]

    for row, cls in enumerate(classes_to_show):
        available = class_map.get(cls, [])
        if not available:
            continue
        paths = rng.sample(available, min(n_per_class, len(available)))
        for col in range(n_cols):
            ax = axes[row][col]
            if col < len(paths):
                img = np.array(Image.open(paths[col]).convert("RGB"))
                ax.imshow(img)
            else:
                ax.axis("off")
            ax.set_xticks([])
            ax.set_yticks([])
            if col == 0:
                short = cls.replace("___", "\n").replace("_", " ")
                ax.set_ylabel(short, fontsize=7, rotation=0, labelpad=60, va="center")

    fig.suptitle("Sample Images per Class", fontsize=14, fontweight="bold")
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig


# ── 4. Image size distribution ───────────────────────────────────────────────
def analyze_image_sizes(
    class_map: Dict[str, List[Path]],
    sample_per_class: int = 30,
    seed: int = 42,
) -> Dict:
    """
    Sample images and return width/height statistics.
    """
    rng = random.Random(seed)
    widths, heights, aspects = [], [], []

    for paths in class_map.values():
        chosen = rng.sample(paths, min(sample_per_class, len(paths)))
        for p in chosen:
            try:
                with Image.open(p) as img:
                    w, h = img.size
                    widths.append(w)
                    heights.append(h)
                    aspects.append(w / h)
            except Exception:
                pass

    return {
        "widths": np.array(widths),
        "heights": np.array(heights),
        "aspects": np.array(aspects),
        "unique_sizes": len(set(zip(widths, heights))),
        "most_common_size": _most_common(list(zip(widths, heights))),
    }


def _most_common(lst):
    from collections import Counter
    if not lst:
        return None
    return Counter(lst).most_common(1)[0][0]


def plot_size_distribution(
    size_stats: Dict,
    figsize: Tuple[int, int] = (14, 4),
    save_path: Optional[str] = None,
) -> plt.Figure:
    """Histograms of widths, heights, and aspect ratios."""
    fig, axes = plt.subplots(1, 3, figsize=figsize)
    pairs = [
        (size_stats["widths"], "Width (px)", "#4CAF50"),
        (size_stats["heights"], "Height (px)", "#2196F3"),
        (size_stats["aspects"], "Aspect Ratio (W/H)", "#FF9800"),
    ]
    for ax, (data, label, color) in zip(axes, pairs):
        ax.hist(data, bins=30, color=color, edgecolor="white", linewidth=0.5)
        ax.set_xlabel(label, fontsize=10)
        ax.set_ylabel("Count", fontsize=10)
        ax.axvline(np.mean(data), color="red", linestyle="--", linewidth=1.2,
                   label=f"Mean={np.mean(data):.1f}")
        ax.legend(fontsize=8)
        ax.spines[["top", "right"]].set_visible(False)

    fig.suptitle("Image Size Distribution (Sampled)", fontsize=13, fontweight="bold")
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig


# ── 5. Pixel intensity / channel statistics ──────────────────────────────────
def compute_channel_stats(
    class_map: Dict[str, List[Path]],
    sample_per_class: int = 20,
    seed: int = 42,
) -> Dict:
    """
    Compute per-channel (R, G, B) mean and std across a random sample.
    """
    rng = random.Random(seed)
    channel_means = [[], [], []]
    channel_stds = [[], [], []]

    for paths in class_map.values():
        chosen = rng.sample(paths, min(sample_per_class, len(paths)))
        for p in chosen:
            try:
                arr = np.array(Image.open(p).convert("RGB"), dtype=np.float32)
                for c in range(3):
                    channel_means[c].append(arr[:, :, c].mean())
                    channel_stds[c].append(arr[:, :, c].std())
            except Exception:
                pass

    result = {}
    for c, name in enumerate(["R", "G", "B"]):
        result[name] = {
            "mean": float(np.mean(channel_means[c])),
            "std": float(np.mean(channel_stds[c])),
        }
    return result


def plot_channel_stats(
    channel_stats: Dict,
    figsize: Tuple[int, int] = (8, 4),
    save_path: Optional[str] = None,
) -> plt.Figure:
    """Bar chart of per-channel mean ± std."""
    channels = ["R", "G", "B"]
    means = [channel_stats[c]["mean"] for c in channels]
    stds = [channel_stats[c]["std"] for c in channels]
    colors = ["#e53935", "#43a047", "#1e88e5"]

    fig, ax = plt.subplots(figsize=figsize)
    x = np.arange(3)
    ax.bar(x, means, yerr=stds, color=colors, width=0.5,
           capsize=8, edgecolor="white", linewidth=0.5)
    ax.set_xticks(x)
    ax.set_xticklabels(["Red", "Green", "Blue"], fontsize=11)
    ax.set_ylabel("Pixel Value (0–255)", fontsize=10)
    ax.set_title("Mean ± Std per Channel (Sampled)", fontsize=12, fontweight="bold")
    ax.set_ylim(0, 280)
    ax.spines[["top", "right"]].set_visible(False)
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig


# ── 6. Healthy vs diseased ratio ─────────────────────────────────────────────
def compute_health_ratio(class_counts: Dict[str, int]) -> Tuple[int, int]:
    """Return (healthy_count, diseased_count) by checking for 'healthy' in class name."""
    healthy = sum(v for k, v in class_counts.items() if "healthy" in k.lower())
    diseased = sum(v for k, v in class_counts.items() if "healthy" not in k.lower())
    return healthy, diseased


def plot_health_ratio(
    class_counts: Dict[str, int],
    figsize: Tuple[int, int] = (5, 5),
    save_path: Optional[str] = None,
) -> plt.Figure:
    """Donut chart: healthy vs diseased."""
    healthy, diseased = compute_health_ratio(class_counts)
    fig, ax = plt.subplots(figsize=figsize)
    wedges, texts, autotexts = ax.pie(
        [healthy, diseased],
        labels=["Healthy", "Diseased"],
        colors=["#66BB6A", "#EF5350"],
        autopct="%1.1f%%",
        startangle=90,
        wedgeprops=dict(width=0.55),
        textprops={"fontsize": 12},
    )
    ax.set_title("Healthy vs Diseased", fontsize=13, fontweight="bold")
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig
