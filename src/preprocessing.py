"""
preprocessing.py
================
Image preprocessing pipeline for PlantVillage:
  - Resizing / padding
  - Normalisation
  - Image quality enhancement (CLAHE, denoising)
  - Before/after visualisation helpers
"""

from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
from PIL import Image, ImageFilter, ImageEnhance
import cv2
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec


# ── Constants ──────────────────────────────────────────────────────────────
DEFAULT_TARGET_SIZE: Tuple[int, int] = (256, 256)   # (width, height)
DEFAULT_RESIZE_MODE: str = "crop"                   # aspect-ratio-safe default
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32)
MODEL_INPUT_SIZE: Tuple[int, int] = (224, 224)

# ── 1. Resizing ──────────────────────────────────────────────────────────────
def resize_image(
    img: np.ndarray,
    target_size: Tuple[int, int] = DEFAULT_TARGET_SIZE,
    mode: str = "crop",
    pad_color: Tuple[int, int, int] = (0, 0, 0),
) -> np.ndarray:
    """
    Resize an RGB image (H×W×3 numpy array).

    Parameters
    ----------
    img : np.ndarray  RGB image (uint8 or float32)
    target_size : (width, height)
    mode : 'stretch' | 'fit' | 'crop'
        - 'stretch': simple resize ignoring aspect ratio
        - 'fit'    : preserve aspect ratio, pad with pad_color
        - 'crop'   : center-crop to target after scaling shortest edge
    pad_color : RGB fill colour for 'fit' mode

    Returns
    -------
    np.ndarray  resized image (uint8)
    """
    pil_img = Image.fromarray(img.astype(np.uint8))
    tw, th = target_size

    if mode == "stretch":
        out = pil_img.resize((tw, th), Image.LANCZOS)

    elif mode == "fit":
        pil_img.thumbnail((tw, th), Image.LANCZOS)
        canvas = Image.new("RGB", (tw, th), pad_color)
        offset = ((tw - pil_img.width) // 2, (th - pil_img.height) // 2)
        canvas.paste(pil_img, offset)
        out = canvas

    elif mode == "crop":
        scale = max(tw / pil_img.width, th / pil_img.height)
        new_w = int(pil_img.width * scale)
        new_h = int(pil_img.height * scale)
        pil_img = pil_img.resize((new_w, new_h), Image.LANCZOS)
        left = (new_w - tw) // 2
        top  = (new_h - th) // 2
        out = pil_img.crop((left, top, left + tw, top + th))

    else:
        raise ValueError(f"Unknown resize mode: {mode!r}")

    return np.array(out, dtype=np.uint8)


def batch_resize(
    class_map: Dict[str, List[Path]],
    target_size: Tuple[int, int] = DEFAULT_TARGET_SIZE,
    mode: str = "stretch",
    output_dir: Optional[str] = None,
    sample_limit: Optional[int] = None,
) -> Dict[str, List[np.ndarray]]:
    """
    Resize all (or *sample_limit*) images in class_map.

    If *output_dir* is given, saves resized images as PNGs.
    Always returns a dict of numpy arrays (in memory).
    """
    result: Dict[str, List[np.ndarray]] = {}
    for cls, paths in class_map.items():
        chosen = paths[:sample_limit] if sample_limit else paths
        resized = []
        for p in chosen:
            try:
                arr = np.array(Image.open(p).convert("RGB"))
                arr_r = resize_image(arr, target_size, mode)
                resized.append(arr_r)
                if output_dir:
                    out_dir = Path(output_dir) / cls
                    out_dir.mkdir(parents=True, exist_ok=True)
                    Image.fromarray(arr_r).save(out_dir / (p.stem + ".png"))
            except Exception as exc:
                import warnings
                warnings.warn(f"[batch_resize] Failed on {p}: {exc}")
        result[cls] = resized
    return result


# ── 2. Normalisation ─────────────────────────────────────────────────────────
def normalize_imagenet(img: np.ndarray) -> np.ndarray:
    """
    Normalize a uint8 RGB image using ImageNet mean/std.
    Output dtype: float32, range roughly [-2.1, 2.6].
    """
    arr = img.astype(np.float32) / 255.0
    return (arr - IMAGENET_MEAN) / IMAGENET_STD


def normalize_minmax(img: np.ndarray) -> np.ndarray:
    """Scale pixel values to [0, 1]."""
    return img.astype(np.float32) / 255.0


# ── 3. Image enhancement / filtering ────────────────────────────────────────
def apply_clahe(
    img: np.ndarray,
    clip_limit: float = 2.0,
    tile_grid_size: Tuple[int, int] = (4, 4),
) -> np.ndarray:
    """
    Apply CLAHE (Contrast Limited Adaptive Histogram Equalisation)
    on the L channel of LAB colour space.

    Parameters
    ----------
    img : uint8 RGB numpy array

    Returns
    -------
    uint8 RGB numpy array with enhanced contrast.
    """
    lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
    l_enhanced = clahe.apply(l)
    lab_enhanced = cv2.merge([l_enhanced, a, b])
    return cv2.cvtColor(lab_enhanced, cv2.COLOR_LAB2RGB)


def apply_gaussian_denoise(img: np.ndarray, kernel_size: int = 3) -> np.ndarray:
    """
    Apply Gaussian blur to reduce noise.

    Parameters
    ----------
    kernel_size : must be odd; higher = more blur.
    """
    if kernel_size % 2 == 0:
        kernel_size += 1
    blurred = cv2.GaussianBlur(img, (kernel_size, kernel_size), 0)
    return blurred.astype(np.uint8)


def apply_bilateral_denoise(
    img: np.ndarray,
    d: int = 5,
    sigma_color: float = 35,
    sigma_space: float = 35,
) -> np.ndarray:
    """
    Edge-preserving bilateral filter — better for natural images.
    """
    bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    filtered = cv2.bilateralFilter(bgr, d, sigma_color, sigma_space)
    return cv2.cvtColor(filtered, cv2.COLOR_BGR2RGB)


def extract_mask_from_segmented(
    segmented_img: np.ndarray,
    black_threshold: int = 15,
) -> np.ndarray:
    """
    Derive a binary foreground mask from a pre-segmented image
    (black background, leaf in color).

    Parameters
    ----------
    segmented_img : uint8 RGB array with black background
    black_threshold : pixels with all channels below this are considered background

    Returns
    -------
    uint8 binary mask (255 = foreground leaf, 0 = background)
    """
    dark = np.all(segmented_img < black_threshold, axis=2)
    mask = np.where(dark, 0, 255).astype(np.uint8)

    # Morphological cleanup — fill small holes inside the leaf
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    return mask


def apply_background_removal(
    img: np.ndarray,
    mask: Optional[np.ndarray] = None,
    bg_color: str = "blur",
) -> np.ndarray:
    """
    Remove background from a leaf image using a precomputed mask or
    HSV-based fallback if no mask is provided.

    Parameters
    ----------
    img : uint8 RGB numpy array
    mask : uint8 binary mask (255=leaf, 0=background). If None, falls back
           to HSV color segmentation (no GrabCut).
    bg_color : 'blur' | 'median' | 'white' | 'black'
        How to fill the background region.
        - 'blur'   : gaussian-blurred version of original (most natural)
        - 'median' : flat median color of the image
        - 'white'  : pure white
        - 'black'  : pure black (avoid for CNNs — creates false edges)

    Returns
    -------
    uint8 RGB numpy array
    """
    h, w = img.shape[:2]

    if mask is None:
        # Fallback: simple HSV color mask, no GrabCut
        bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
        green_mask  = cv2.inRange(hsv, np.array([20, 15, 15]),  np.array([100, 255, 255]))
        brown_mask  = cv2.inRange(hsv, np.array([0,  20, 20]),  np.array([20,  255, 200]))
        yellow_mask = cv2.inRange(hsv, np.array([15, 30, 80]),  np.array([35,  255, 255]))
        mask = cv2.bitwise_or(green_mask, cv2.bitwise_or(brown_mask, yellow_mask))
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=3)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN,  kernel, iterations=1)

    # Build background fill
    if bg_color == "blur":
        background = cv2.GaussianBlur(img, (51, 51), 0)
    elif bg_color == "median":
        med = np.median(img.reshape(-1, 3), axis=0).astype(np.uint8)
        background = np.full_like(img, med)
    elif bg_color == "white":
        background = np.full_like(img, 255)
    else:  # black
        background = np.zeros_like(img)

    # Soft edge: erode mask slightly then blur for feathering
    eroded    = cv2.erode(mask, np.ones((3, 3), np.uint8), iterations=1)
    soft_mask = cv2.GaussianBlur(eroded.astype(np.float32), (9, 9), 0) / 255.0
    soft_mask = soft_mask[:, :, np.newaxis]
    result    = img * soft_mask + background * (1 - soft_mask)
    return result.astype(np.uint8)

def full_enhancement_pipeline(
    img: np.ndarray,
    segmented_img: Optional[np.ndarray] = None,
    bg_color: str = "blur",
) -> np.ndarray:
    """
    Correct pipeline order:
      1. Bilateral denoise    — on real pixels only, before any synthetic fill
      2. CLAHE                — adaptive contrast on real pixels only
      3. Background removal   — synthetic fill pixels never go through steps 1-2
    """
    # Step 1 & 2: enhance real image content first
    img = apply_bilateral_denoise(img)
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    clip = 1.5 if gray.std() > 45 else 2.5
    img = apply_clahe(img, clip_limit=clip)
    # Step 3: background removal last — blurred fill stays unenhanced
    mask = extract_mask_from_segmented(segmented_img) if segmented_img is not None else None
    img = apply_background_removal(img, mask=mask, bg_color=bg_color)
    return img


# ── 4. Before / After visualisation ─────────────────────────────────────────
def plot_before_after(
    original_images: List[np.ndarray],
    processed_images: List[np.ndarray],
    titles: Optional[List[str]] = None,
    step_name: str = "Processing Step",
    figsize_per_img: float = 2.8,
    save_path: Optional[str] = None,
) -> plt.Figure:
    """
    Side-by-side before/after grid.

    Parameters
    ----------
    original_images : list of original RGB arrays
    processed_images : matching list of processed RGB arrays
    titles : optional per-image subtitle list
    step_name : label for the 'After' row
    """
    n = len(original_images)
    fig, axes = plt.subplots(
        2, n,
        figsize=(figsize_per_img * n, figsize_per_img * 2 + 0.5),
    )
    if n == 1:
        axes = [[axes[0]], [axes[1]]]

    row_labels = ["Before", f"After\n({step_name})"]
    for row, (imgs, label) in enumerate(zip([original_images, processed_images], row_labels)):
        for col, img in enumerate(imgs):
            ax = axes[row][col]
            # Convert float images (e.g. ImageNet-normalised or 0-1) back to uint8 for display
            img_disp = img
            if np.issubdtype(img_disp.dtype, np.floating):
                vmin, vmax = float(img_disp.min()), float(img_disp.max())
                if vmin >= 0.0 and vmax <= 1.0:
                    img_disp = img_disp * 255.0
                else:
                    # Assume ImageNet normalisation — invert it for visualization
                    img_disp = (img_disp * IMAGENET_STD.reshape(1, 1, 3)
                                + IMAGENET_MEAN.reshape(1, 1, 3)) * 255.0
                img_disp = np.clip(img_disp, 0, 255).astype(np.uint8)
            else:
                img_disp = img_disp.astype(np.uint8)

            ax.imshow(img_disp)
            ax.set_xticks([])
            ax.set_yticks([])
            if titles and col < len(titles):
                ax.set_title(titles[col], fontsize=8)
        axes[row][0].set_ylabel(label, fontsize=10, rotation=90, labelpad=8, va="center")

    fig.suptitle(f"Before / After — {step_name}", fontsize=13, fontweight="bold")
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig


def plot_pipeline_stages(
    img: np.ndarray,
    figsize: Tuple[int, int] = (18, 4),
    save_path: Optional[str] = None,
) -> plt.Figure:
    """
    Show a single image through every preprocessing stage side-by-side.
    """
    img = img.astype(np.uint8)
    resized    = resize_image(img, DEFAULT_TARGET_SIZE, mode="crop")
    bg_removed = apply_background_removal(resized, bg_color="blur")
    denoised   = apply_bilateral_denoise(bg_removed)
    clahe_img  = apply_clahe(denoised)

    stages = [
        (img,        "Original"),
        (resized,    f"Resized\n{DEFAULT_TARGET_SIZE}"),
        (bg_removed, "Background\nRemoval"),
        (denoised,   "Bilateral\nDenoise"),
        (clahe_img,  "CLAHE"),
    ]

    fig, axes = plt.subplots(1, len(stages), figsize=(22, 4))
    for ax, (stage_img, label) in zip(axes, stages):
        ax.imshow(stage_img)
        ax.set_title(label, fontsize=9, fontweight="bold")
        ax.set_xticks([])
        ax.set_yticks([])

    fig.suptitle("Preprocessing Pipeline — Step by Step", fontsize=13, fontweight="bold")
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig


def plot_histogram_comparison(
    original: np.ndarray,
    processed: np.ndarray,
    save_path: Optional[str] = None,
) -> plt.Figure:
    """
    Compare per-channel histograms of two images.
    """
    fig, axes = plt.subplots(2, 3, figsize=(12, 5))
    channel_names = ["Red", "Green", "Blue"]
    colors = ["#e53935", "#43a047", "#1e88e5"]

    for row, (img, title) in enumerate([(original, "Before"), (processed, "After")]):
        # Prepare image for histogram: convert floats back to uint8 where needed
        img_u8 = img
        if np.issubdtype(img_u8.dtype, np.floating):
            vmin, vmax = float(img_u8.min()), float(img_u8.max())
            if vmin >= 0.0 and vmax <= 1.0:
                img_u8 = (img_u8 * 255.0)
            else:
                img_u8 = (img_u8 * IMAGENET_STD.reshape(1, 1, 3)
                         + IMAGENET_MEAN.reshape(1, 1, 3)) * 255.0
            img_u8 = np.clip(img_u8, 0, 255).astype(np.uint8)
        else:
            # Denormalize if float32 (imagenet-normalized) before plotting
            if img.dtype != np.uint8:
                img_u8 = np.clip(
                    (img * np.array([0.229, 0.224, 0.225]) + np.array([0.485, 0.456, 0.406])) * 255,
                    0, 255
                ).astype(np.uint8)
            else:
                img_u8 = img.astype(np.uint8)

        for col, (ch, cname, color) in enumerate(zip(range(3), channel_names, colors)):
            ax = axes[row][col]
            ax.hist(img_u8[:, :, ch].ravel(), bins=64, color=color,
                    alpha=0.8, edgecolor="none")
            ax.set_xlim(0, 255)
            ax.set_title(f"{title} — {cname}", fontsize=9)
            ax.spines[["top", "right"]].set_visible(False)

    fig.suptitle("Pixel Intensity Distributions — Before vs After", fontsize=12, fontweight="bold")
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig
