#!/usr/bin/env python3
"""
TTI & GNG Overlap Analysis - Project ii
Compares Tool-Tissue Interaction (TTI) detections with Go/No-Go zone segmentations
to analyze differences between safe and BDI (Bile Duct Injury) laparoscopic videos.

Main Research Question: Can an AI model detect differences in tool-tissue interactions
between go and no-go zones across safe and BDI videos?

Expected Output:
- Per-frame CSV files with overlap statistics
- Per-video summary CSV files
- Combined summary CSV comparing safe vs BDI videos
- Visualization overlays (PNG images)
- Statistical comparison results
- Bar charts showing zone distribution

Estimated completion: October 3rd
"""

import os
import zipfile
from multiprocessing import Pool
from typing import Dict, Optional, Tuple

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

# =============================================================================
# CONFIGURATION SECTION - Edit these settings as needed
# =============================================================================

# -----------------------------------------------------------------------------
# HSV Color Thresholds for Mask Detection
# -----------------------------------------------------------------------------
# TTI (Tool-Tissue Interaction) - neon green segmentation in TTI videos
TTI_HSV_LO = np.array([40, 120, 80])
TTI_HSV_HI = np.array([85, 255, 255])

# Go Zone - green regions in zone-map videos
GO_HSV_LO = np.array([40, 40, 40])
GO_HSV_HI = np.array([85, 255, 255])

# No-Go Zone - red regions in zone-map videos (split for hue wrapping)
RED1_LO = np.array([0, 70, 50])  # Low hue red
RED1_HI = np.array([10, 255, 255])
RED2_LO = np.array([170, 70, 50])  # High hue red
RED2_HI = np.array([180, 255, 255])

# Unclear Zone - if your GNGNet outputs unclear zones, add HSV ranges here
# UNCLEAR_HSV_LO = np.array([?, ?, ?])
# UNCLEAR_HSV_HI = np.array([?, ?, ?])
USE_UNCLEAR_ZONE = False  # Set to True if you have unclear zone data

# -----------------------------------------------------------------------------
# Video Processing Settings
# -----------------------------------------------------------------------------
INTERVAL_SECONDS = 1  # Sample every N seconds from videos
SEARCH_WINDOW_S = 0.25  # Window to search for best TTI frame (±seconds)
SEARCH_STEP_S = 0.05  # Step size when searching for best frame
NUM_CPUS = 6  # Number of parallel workers for video processing

# -----------------------------------------------------------------------------
# Synchronization Settings
# -----------------------------------------------------------------------------
# Offset to align zone-map video with TTI video (in seconds)
# Positive values shift zone-map forward in time
DEFAULT_SYNC_OFFSET_S = 0.3

# Explanation: At 30 fps, 0.3s ≈ 9 frames
# This compensates for video processing/export delays between TTI and zone videos
# Adjust per video if needed - see video_pairs configuration below

# -----------------------------------------------------------------------------
# Mask Filtering Settings
# -----------------------------------------------------------------------------
MIN_TTI_AREA = 250  # Minimum pixel area for valid TTI detection
MAX_TTI_ASPECT_RATIO = 2.0  # Maximum width/height ratio for TTI blobs

# -----------------------------------------------------------------------------
# Visualization Settings
# -----------------------------------------------------------------------------
OUTLINE_COLOR = (255, 255, 255)  # White outline for TTI contours
OUTLINE_THICKNESS = 3
FONT_SCALE = 0.9
FONT_THICKNESS = 2
FONT = cv2.FONT_HERSHEY_SIMPLEX

# Go/No-Go visualization colors (BGR format)
VIS_GO_COLOR = (0, 255, 0)  # Green
VIS_NOGO_COLOR = (0, 0, 255)  # Red
VIS_BG_COLOR = (0, 0, 0)  # Black

# -----------------------------------------------------------------------------
# Output Settings
# -----------------------------------------------------------------------------
BASE_OUTPUT_DIR = (
    "/cluster/projects/madanigroup/lorenz/tti/unified_fps_tti_gng_overlay_analysis"
)
GENERATE_ZIPS = True  # Create ZIP files of outputs for easy sharing
SAVE_VISUALIZATIONS = True  # Save PNG overlay images
SAVE_PER_FRAME_CSV = True  # Save detailed per-frame statistics
SAVE_VIDEO_SUMMARY_CSV = True  # Save per-video summary statistics

# -----------------------------------------------------------------------------
# Statistical Analysis Settings
# -----------------------------------------------------------------------------
SIGNIFICANCE_LEVEL = 0.05  # Alpha level for statistical tests
USE_TTEST = True  # True for t-test, False for Mann-Whitney U test

# -----------------------------------------------------------------------------
# Video Pairs Configuration
# Format: (tti_video_path, zone_video_path, sync_offset_seconds, video_type, case_id)
# video_type should be either "safe" or "BDI"
# -----------------------------------------------------------------------------

VIDEO_PAIRS = [
    # Safe Laparoscopic Cholecystectomy Videos (28 total)
    (
        "/cluster/projects/madanigroup/lorenz/tti/videos/LapChol Case 0001 03.mp4",
        "/cluster/projects/madanigroup/lorenz/tti/gng_predictions_no_background/LapChol Case 0001 03.MP4",
        0.3,
        "safe",
        "lapchol_0001_03",
    ),
    (
        "/cluster/projects/madanigroup/lorenz/tti/videos/LapChol Case 0001 04.mp4",
        "/cluster/projects/madanigroup/lorenz/tti/gng_predictions_no_background/LapChol Case 0001 04.MP4",
        0.3,
        "safe",
        "lapchol_0001_04",
    ),
    (
        "/cluster/projects/madanigroup/lorenz/tti/videos/LapChol Case 0001 05.mp4",
        "/cluster/projects/madanigroup/lorenz/tti/gng_predictions_no_background/LapChol Case 0001 05.MP4",
        0.3,
        "safe",
        "lapchol_0001_05",
    ),
    (
        "/cluster/projects/madanigroup/lorenz/tti/videos/LapChol Case 0002 02.mp4",
        "/cluster/projects/madanigroup/lorenz/tti/gng_predictions_no_background/LapChol Case 0002 02.MP4",
        0.3,
        "safe",
        "lapchol_0002_02",
    ),
    (
        "/cluster/projects/madanigroup/lorenz/tti/videos/LapChol Case 0002 03.mp4",
        "/cluster/projects/madanigroup/lorenz/tti/gng_predictions_no_background/LapChol Case 0002 03.MP4",
        0.3,
        "safe",
        "lapchol_0002_03",
    ),
    (
        "/cluster/projects/madanigroup/lorenz/tti/videos/LapChol Case 0007 01.mp4",
        "/cluster/projects/madanigroup/lorenz/tti/gng_predictions_no_background/LapChol Case 0007 01.MP4",
        0.3,
        "safe",
        "lapchol_0007_01",
    ),
    (
        "/cluster/projects/madanigroup/lorenz/tti/videos/LapChol Case 0007 02.mp4",
        "/cluster/projects/madanigroup/lorenz/tti/gng_predictions_no_background/LapChol Case 0007 02.MP4",
        0.3,
        "safe",
        "lapchol_0007_02",
    ),
    (
        "/cluster/projects/madanigroup/lorenz/tti/videos/LapChol Case 0007 03.mp4",
        "/cluster/projects/madanigroup/lorenz/tti/gng_predictions_no_background/LapChol Case 0007 03.MP4",
        0.3,
        "safe",
        "lapchol_0007_03",
    ),
    (
        "/cluster/projects/madanigroup/lorenz/tti/videos/LapChol Case 0011 02.mp4",
        "/cluster/projects/madanigroup/lorenz/tti/gng_predictions_no_background/LapChol Case 0011 02.MP4",
        0.3,
        "safe",
        "lapchol_0011_02",
    ),
    (
        "/cluster/projects/madanigroup/lorenz/tti/videos/LapChol Case 0011 03.mp4",
        "/cluster/projects/madanigroup/lorenz/tti/gng_predictions_no_background/LapChol Case 0011 03.MP4",
        0.3,
        "safe",
        "lapchol_0011_03",
    ),
    (
        "/cluster/projects/madanigroup/lorenz/tti/videos/LapChol Case 0012 03.mp4",
        "/cluster/projects/madanigroup/lorenz/tti/gng_predictions_no_background/LapChol Case 0012 03.MP4",
        0.3,
        "safe",
        "lapchol_0012_03",
    ),
    (
        "/cluster/projects/madanigroup/lorenz/tti/videos/LapChol Case 0012 04.mp4",
        "/cluster/projects/madanigroup/lorenz/tti/gng_predictions_no_background/LapChol Case 0012 04.MP4",
        0.3,
        "safe",
        "lapchol_0012_04",
    ),
    (
        "/cluster/projects/madanigroup/lorenz/tti/videos/LapChol Case 0015 01.mp4",
        "/cluster/projects/madanigroup/lorenz/tti/gng_predictions_no_background/LapChol Case 0015 01.MP4",
        0.3,
        "safe",
        "lapchol_0015_01",
    ),
    (
        "/cluster/projects/madanigroup/lorenz/tti/videos/LapChol Case 0015 02.mp4",
        "/cluster/projects/madanigroup/lorenz/tti/gng_predictions_no_background/LapChol Case 0015 02.MP4",
        0.3,
        "safe",
        "lapchol_0015_02",
    ),
    (
        "/cluster/projects/madanigroup/lorenz/tti/videos/LapChol Case 0016 01.mp4",
        "/cluster/projects/madanigroup/lorenz/tti/gng_predictions_no_background/LapChol Case 0016 01.MP4",
        0.3,
        "safe",
        "lapchol_0016_01",
    ),
    (
        "/cluster/projects/madanigroup/lorenz/tti/videos/LapChol Case 0018 10.mp4",
        "/cluster/projects/madanigroup/lorenz/tti/gng_predictions_no_background/LapChol Case 0018 10.MP4",
        0.3,
        "safe",
        "lapchol_0018_10",
    ),
    (
        "/cluster/projects/madanigroup/lorenz/tti/videos/LapChol Case 0018 11.mp4",
        "/cluster/projects/madanigroup/lorenz/tti/gng_predictions_no_background/LapChol Case 0018 11.MP4",
        0.3,
        "safe",
        "lapchol_0018_11",
    ),
    (
        "/cluster/projects/madanigroup/lorenz/tti/videos/LapChol Case 0019 02.mp4",
        "/cluster/projects/madanigroup/lorenz/tti/gng_predictions_no_background/LapChol Case 0019 02.MP4",
        0.3,
        "safe",
        "lapchol_0019_02",
    ),
    (
        "/cluster/projects/madanigroup/lorenz/tti/videos/LapChol Case 0019 03.mp4",
        "/cluster/projects/madanigroup/lorenz/tti/gng_predictions_no_background/LapChol Case 0019 03.MP4",
        0.3,
        "safe",
        "lapchol_0019_03",
    ),
    (
        "/cluster/projects/madanigroup/lorenz/tti/videos/LapChol Case 0020 02.mp4",
        "/cluster/projects/madanigroup/lorenz/tti/gng_predictions_no_background/LapChol Case 0020 02.MP4",
        0.3,
        "safe",
        "lapchol_0020_02",
    ),
    (
        "/cluster/projects/madanigroup/lorenz/tti/videos/LapChol Case 0020 03.mp4",
        "/cluster/projects/madanigroup/lorenz/tti/gng_predictions_no_background/LapChol Case 0020 03.MP4",
        0.3,
        "safe",
        "lapchol_0020_03",
    ),
    (
        "/cluster/projects/madanigroup/lorenz/tti/videos/LapChol Case 0023 03.mp4",
        "/cluster/projects/madanigroup/lorenz/tti/gng_predictions_no_background/LapChol Case 0023 03.mp4",
        0.3,
        "safe",
        "lapchol_0023_03",
    ),
    (
        "/cluster/projects/madanigroup/lorenz/tti/videos/LapChol Case 0023 04.mp4",
        "/cluster/projects/madanigroup/lorenz/tti/gng_predictions_no_background/LapChol Case 0023 04.MP4",
        0.3,
        "safe",
        "lapchol_0023_04",
    ),
    # BDI (Bile Duct Injury) Videos (11 total)
    (
        "/cluster/projects/madanigroup/lorenz/tti/videos/V10-Trimmed.mp4",
        "/cluster/projects/madanigroup/lorenz/tti/gng_predictions_no_background/V10-Trimmed.mp4",
        0.3,
        "BDI",
        "V10_trimmed",
    ),
    (
        "/cluster/projects/madanigroup/lorenz/tti/videos/V11-Trimmed.mp4",
        "/cluster/projects/madanigroup/lorenz/tti/gng_predictions_no_background/V11-Trimmed.mov",
        0.3,
        "BDI",
        "V11_trimmed",
    ),
    (
        "/cluster/projects/madanigroup/lorenz/tti/videos/V12-Trimmed.mp4",
        "/cluster/projects/madanigroup/lorenz/tti/gng_predictions_no_background/V12-Trimmed.mp4",
        0.3,
        "BDI",
        "V12_trimmed",
    ),
    (
        "/cluster/projects/madanigroup/lorenz/tti/videos/V14_Trimmed.mp4",
        "/cluster/projects/madanigroup/lorenz/tti/gng_predictions_no_background/V14_Trimmed.mp4",
        0.3,
        "BDI",
        "V14_trimmed",
    ),
    (
        "/cluster/projects/madanigroup/lorenz/tti/videos/V15_Trimmed.mp4",
        "/cluster/projects/madanigroup/lorenz/tti/gng_predictions_no_background/V15_Trimmed.mp4",
        0.3,
        "BDI",
        "V15_trimmed",
    ),
    (
        "/cluster/projects/madanigroup/lorenz/tti/videos/V17_Trimmed.mp4",
        "/cluster/projects/madanigroup/lorenz/tti/gng_predictions_no_background/V17_Trimmed.mp4",
        0.3,
        "BDI",
        "V17_trimmed",
    ),
    (
        "/cluster/projects/madanigroup/lorenz/tti/videos/V18_Trimmed.mp4",
        "/cluster/projects/madanigroup/lorenz/tti/gng_predictions_no_background/V18_Trimmed.mp4",
        0.3,
        "BDI",
        "V18_trimmed",
    ),
    (
        "/cluster/projects/madanigroup/lorenz/tti/videos/V2_Trimmed.mp4",
        "/cluster/projects/madanigroup/lorenz/tti/gng_predictions_no_background/V2_Trimmed.mp4",
        0.3,
        "BDI",
        "V2_trimmed",
    ),
    (
        "/cluster/projects/madanigroup/lorenz/tti/videos/V4_Trimmed.mp4",
        "/cluster/projects/madanigroup/lorenz/tti/gng_predictions_no_background/V4_Trimmed.mp4",
        0.3,
        "BDI",
        "V4_trimmed",
    ),
    (
        "/cluster/projects/madanigroup/lorenz/tti/videos/V5_Trimmed.mp4",
        "/cluster/projects/madanigroup/lorenz/tti/gng_predictions_no_background/V5_Trimmed.mp4",
        0.3,
        "BDI",
        "V5_trimmed",
    ),
    (
        "/cluster/projects/madanigroup/lorenz/tti/videos/V7-Trimmed.mp4",
        "/cluster/projects/madanigroup/lorenz/tti/gng_predictions_no_background/V7-Trimmed.mp4",
        0.3,
        "BDI",
        "V7_trimmed",
    ),
]

# =============================================================================
# END OF CONFIGURATION SECTION
# =============================================================================


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================


def largest_reasonable_green_component(
    mask: np.ndarray,
    min_area: int = MIN_TTI_AREA,
    max_aspect: float = MAX_TTI_ASPECT_RATIO,
) -> np.ndarray:
    """
    Select the largest valid green mask region based on area and aspect ratio.

    Args:
        mask: Binary mask image
        min_area: Minimum contour area to consider
        max_aspect: Maximum width/height aspect ratio

    Returns:
        Cleaned binary mask with only the largest valid component
    """
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    best = None
    best_area = 0

    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < min_area:
            continue

        x, y, w, h = cv2.boundingRect(cnt)
        aspect = w / float(h + 1e-6)
        if aspect > max_aspect:
            continue

        if area > best_area:
            best_area = area
            best = cnt

    clean = np.zeros(mask.shape, dtype=np.uint8)
    if best is not None:
        cv2.drawContours(clean, [best], -1, 255, thickness=cv2.FILLED)

    return clean


def set_time(
    cap: cv2.VideoCapture, t_seconds: float
) -> Tuple[bool, Optional[np.ndarray]]:
    """
    Seek to approximately t_seconds in video using milliseconds.

    Args:
        cap: OpenCV VideoCapture object
        t_seconds: Target time in seconds

    Returns:
        Tuple of (success, frame) where frame is None if unsuccessful
    """
    if t_seconds < 0:
        return False, None

    cap.set(cv2.CAP_PROP_POS_MSEC, float(t_seconds) * 1000.0)
    ok, frame = cap.read()
    return ok, frame


def tti_mask_from_frame(frame_bgr: np.ndarray) -> np.ndarray:
    """
    Create cleaned TTI mask from a TTI-video frame using HSV thresholds.

    Args:
        frame_bgr: Input frame in BGR color space

    Returns:
        Binary mask with TTI region
    """
    hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)
    raw = cv2.inRange(hsv, TTI_HSV_LO, TTI_HSV_HI)
    raw = cv2.medianBlur(raw, 5)
    return largest_reasonable_green_component(raw)


def zone_masks_from_frame(
    frame_bgr: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Extract Go/No-Go/Background zone masks from a zone-map frame.

    Args:
        frame_bgr: Input frame in BGR color space

    Returns:
        Tuple of (go_mask, nogo_mask, bg_mask) as binary masks
    """
    hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)

    # Go zone (green)
    go = cv2.inRange(hsv, GO_HSV_LO, GO_HSV_HI)

    # No-Go zone (red - handle hue wrapping)
    r1 = cv2.inRange(hsv, RED1_LO, RED1_HI)
    r2 = cv2.inRange(hsv, RED2_LO, RED2_HI)
    nogo = cv2.bitwise_or(r1, r2)

    # Background (everything else)
    bg = cv2.bitwise_not(cv2.bitwise_or(go, nogo))

    # TODO: Add unclear zone detection if USE_UNCLEAR_ZONE is True
    # unclear = cv2.inRange(hsv, UNCLEAR_HSV_LO, UNCLEAR_HSV_HI)

    return go, nogo, bg


def find_best_tti_frame(
    cap_tti: cv2.VideoCapture,
    t_guess: float,
    search_window_s: float = SEARCH_WINDOW_S,
    step_s: float = SEARCH_STEP_S,
) -> Tuple[Optional[float], Optional[np.ndarray], Optional[np.ndarray], int]:
    """
    Search around t_guess ± search_window_s to find the frame with largest TTI blob.
    This helps ensure we're measuring the clearest/most representative interaction.

    Args:
        cap_tti: VideoCapture object for TTI video
        t_guess: Initial time guess in seconds
        search_window_s: Search window size (± seconds)
        step_s: Time step for searching

    Returns:
        Tuple of (best_time, best_frame_bgr, best_mask, best_area)
        Returns (None, None, None, 0) if no valid frame found
    """
    candidates = np.arange(
        t_guess - search_window_s, t_guess + search_window_s + 1e-6, step_s
    )

    best = (None, None, None, 0)

    for t in candidates:
        ok, frame = set_time(cap_tti, t)
        if not ok or frame is None:
            continue

        mask = tti_mask_from_frame(frame)
        area = int(np.count_nonzero(mask))

        if area > best[3]:
            best = (t, frame, mask, area)

    return best


def make_video_zips(video_out_dir: str) -> Tuple[Optional[str], Optional[str]]:
    """
    Create two ZIP files for a video's outputs:
    1. <folder>_images.zip: All PNG/JPG overlay images
    2. <folder>_csvs.zip: All CSV files

    Args:
        video_out_dir: Directory containing video outputs

    Returns:
        Tuple of (images_zip_path, csvs_zip_path)
        Returns None for either if no files of that type exist
    """
    if not GENERATE_ZIPS:
        return None, None

    folder_name = os.path.basename(os.path.normpath(video_out_dir))
    images_zip = os.path.join(video_out_dir, f"{folder_name}_images.zip")
    csvs_zip = os.path.join(video_out_dir, f"{folder_name}_csvs.zip")

    # Collect files by extension
    image_exts = {".png", ".jpg", ".jpeg"}
    csv_exts = {".csv"}

    image_files, csv_files = [], []

    for fn in os.listdir(video_out_dir):
        fpath = os.path.join(video_out_dir, fn)
        if not os.path.isfile(fpath):
            continue

        ext = os.path.splitext(fn)[1].lower()
        if ext in image_exts:
            image_files.append(fpath)
        elif ext in csv_exts:
            csv_files.append(fpath)

    # Create ZIP files only if there are files
    if image_files:
        with zipfile.ZipFile(images_zip, "w", compression=zipfile.ZIP_DEFLATED) as zf:
            for f in sorted(image_files):
                zf.write(f, arcname=os.path.basename(f))
    else:
        images_zip = None

    if csv_files:
        with zipfile.ZipFile(csvs_zip, "w", compression=zipfile.ZIP_DEFLATED) as zf:
            for f in sorted(csv_files):
                zf.write(f, arcname=os.path.basename(f))
    else:
        csvs_zip = None

    return images_zip, csvs_zip


# =============================================================================
# MAIN ANALYSIS FUNCTIONS
# =============================================================================


def compute_overlap_stats(
    tti_video_path: str,
    zonemap_video_path: str,
    out_dir: str,
    video_type: str,
    case_id: str,
    interval_seconds: int = INTERVAL_SECONDS,
    sync_offset_s: float = DEFAULT_SYNC_OFFSET_S,
) -> Tuple[Optional[str], str, str, Dict]:
    """
    Main analysis function: Compute TTI & GNG zone overlap statistics for a video pair.

    Process:
    1. For each whole second:
       - Find TTI frame with largest blob within ±search_window
       - Sample zone-map at aligned time (best_tti_time + sync_offset)
       - Compute Go/No-Go/BG percentages inside TTI mask
    2. Save per-frame CSV with detailed statistics
    3. Save one-row summary CSV with video-level aggregates
    4. Generate visualization overlays (PNG)

    Args:
        tti_video_path: Path to TTI detection video
        zonemap_video_path: Path to zone-map video
        out_dir: Base output directory
        video_type: "safe" or "BDI"
        case_id: Unique identifier for this case
        interval_seconds: Sample every N seconds
        sync_offset_s: Time offset to align videos

    Returns:
        Tuple of (example_png_path, summary_csv_path, video_out_dir, video_stats_dict)
    """
    os.makedirs(out_dir, exist_ok=True)

    # Open video captures
    cap_tti = cv2.VideoCapture(tti_video_path)
    cap_zone = cv2.VideoCapture(zonemap_video_path)

    if not cap_tti.isOpened() or not cap_zone.isOpened():
        raise FileNotFoundError(
            f"Cannot open videos:\n"
            f"  TTI : {tti_video_path}\n"
            f"  ZONE: {zonemap_video_path}"
        )

    # Get first frame for dimensions
    ok0, frame0 = set_time(cap_tti, 0.0)
    if not ok0:
        raise ValueError("Failed to read first frame of TTI video.")
    H, W = frame0.shape[:2]

    # Create output directory for this video
    stem = os.path.splitext(os.path.basename(tti_video_path))[0]
    video_out_dir = os.path.join(out_dir, f"{stem}__trained")
    os.makedirs(video_out_dir, exist_ok=True)

    # Calculate video duration (use shorter of the two videos)
    n_tti = int(cap_tti.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    n_zone = int(cap_zone.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    fps_tti = cap_tti.get(cv2.CAP_PROP_FPS) or 30.0
    fps_zone = cap_zone.get(cv2.CAP_PROP_FPS) or 30.0
    dur_tti = (n_tti / max(fps_tti, 1.0)) if n_tti else 0.0
    dur_zone = (n_zone / max(fps_zone, 1.0)) if n_zone else 0.0
    duration = max(0, int(min(dur_tti, dur_zone)))

    rows = []
    example_png = None

    # Process each second of video
    for sec in range(0, duration + 1, interval_seconds):
        # Find best TTI frame near this second
        best_t, tti_bgr, tti_mask, area = find_best_tti_frame(
            cap_tti, float(sec), search_window_s=SEARCH_WINDOW_S, step_s=SEARCH_STEP_S
        )

        if area == 0 or tti_bgr is None:
            continue

        # Get corresponding zone-map frame (with sync offset)
        zone_time = (best_t if best_t is not None else float(sec)) + float(
            sync_offset_s
        )
        okz, zone_bgr = set_time(cap_zone, zone_time)

        if not okz or zone_bgr is None:
            continue

        # Resize zone frame to match TTI frame if needed
        if zone_bgr.shape[:2] != (H, W):
            zone_bgr = cv2.resize(zone_bgr, (W, H), interpolation=cv2.INTER_NEAREST)

        # Extract zone masks
        go_mask, nogo_mask, bg_mask = zone_masks_from_frame(zone_bgr)

        # Compute overlap statistics
        idx = np.where(tti_mask == 255)
        tti_pixels = int(len(idx[0]))

        if tti_pixels == 0:
            continue

        go_px = int(np.sum(go_mask[idx] == 255))
        nogo_px = int(np.sum(nogo_mask[idx] == 255))
        bg_px = int(np.sum(bg_mask[idx] != 0))

        go_pct = 100.0 * go_px / tti_pixels
        nogo_pct = 100.0 * nogo_px / tti_pixels
        bg_pct = 100.0 * bg_px / tti_pixels

        # Save per-frame statistics
        rows.append(
            {
                "timestamp_s": round(best_t, 3) if best_t is not None else sec,
                "tti_pixels": tti_pixels,
                "go_pixels": go_px,
                "nogo_pixels": nogo_px,
                "background_pixels": bg_px,
                "go_percent": round(go_pct, 4),
                "nogo_percent": round(nogo_pct, 4),
                "background_percent": round(bg_pct, 4),
            }
        )

        # Generate visualization overlay
        if SAVE_VISUALIZATIONS:
            # Create clean background with zone colors
            bg = np.zeros_like(zone_bgr)
            bg[go_mask == 255] = VIS_GO_COLOR
            bg[nogo_mask == 255] = VIS_NOGO_COLOR

            vis = bg.copy()

            # Draw TTI outline (not filled)
            contours, _ = cv2.findContours(
                tti_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
            )
            if contours:
                cv2.drawContours(vis, contours, -1, OUTLINE_COLOR, OUTLINE_THICKNESS)

            # Add text overlay with statistics
            header = f"{int(round(sec))}s | Go {go_pct:.2f}% | No-Go {nogo_pct:.2f}% | BG {bg_pct:.2f}%"
            cv2.putText(
                vis, header, (16, 36), FONT, FONT_SCALE, OUTLINE_COLOR, FONT_THICKNESS
            )

            # Save visualization
            out_png = os.path.join(
                video_out_dir, f"{stem}__{sec:05d}s_trained_outline.png"
            )
            cv2.imwrite(out_png, vis)

            if example_png is None:
                example_png = out_png

    cap_tti.release()
    cap_zone.release()

    # Create DataFrames from collected data
    df = pd.DataFrame(rows)

    if df.empty:
        raise ValueError(
            "No valid TTI detections after alignment. "
            "Try adjusting sync_offset_s or HSV thresholds."
        )

    # Save per-frame CSV
    if SAVE_PER_FRAME_CSV:
        perframe_csv = os.path.join(video_out_dir, f"{stem}__trained_perframe.csv")
        df.to_csv(perframe_csv, index=False)

    # Compute video-level summary statistics
    tti_total = int(df["tti_pixels"].sum())
    go_total = int(df["go_pixels"].sum())
    nogo_total = int(df["nogo_pixels"].sum())
    bg_total = int(df["background_pixels"].sum())

    # Calculate video-level percentages
    go_pct_total = round(100.0 * go_total / max(1, tti_total), 4)
    nogo_pct_total = round(100.0 * nogo_total / max(1, tti_total), 4)
    bg_pct_total = round(100.0 * bg_total / max(1, tti_total), 4)

    # Create summary row
    summary_row = pd.DataFrame(
        [
            {
                "video_type": video_type,
                "case_id": case_id,
                "image": os.path.basename(tti_video_path),
                "tti_pixels": tti_total,
                "go_pixels": go_total,
                "nogo_pixels": nogo_total,
                "background_pixels": bg_total,
                "go_percent": go_pct_total,
                "nogo_percent": nogo_pct_total,
                "background_percent": bg_pct_total,
                "go_plus_unclear_percent": go_pct_total,  # TODO: Add unclear when implemented
                "nogo_plus_unclear_percent": nogo_pct_total,  # TODO: Add unclear when implemented
            }
        ]
    )

    # Save video summary CSV
    summary_csv = os.path.join(video_out_dir, f"{stem}__trained_stats.csv")
    if SAVE_VIDEO_SUMMARY_CSV:
        summary_row.to_csv(summary_csv, index=False)

    # Return statistics dictionary for aggregation
    video_stats = {
        "video_type": video_type,
        "case_id": case_id,
        "go_percent": go_pct_total,
        "nogo_percent": nogo_pct_total,
        "background_percent": bg_pct_total,
        "total_tti_pixels": tti_total,
    }

    return example_png, summary_csv, video_out_dir, video_stats


def generate_comparison_visualizations(combined_df: pd.DataFrame, output_dir: str):
    """
    Generate the 4 visualization types required by Project ii (from slide 8).

    Creates bar charts comparing Safe vs BDI videos:
    1. Proportion NG + unclear vs Proportion go zone interactions
    2. Proportion go zone + unclear vs Proportion go zone
    3. Proportion NG vs Unclear vs Proportion go zone
    4. Proportion of No Go zone vs Proportion go zone

    Args:
        combined_df: DataFrame with all video statistics
        output_dir: Directory to save visualization plots
    """
    os.makedirs(output_dir, exist_ok=True)

    safe_df = combined_df[combined_df["video_type"] == "safe"]
    bdi_df = combined_df[combined_df["video_type"] == "BDI"]

    # Calculate means for each group
    safe_means = {
        "go": safe_df["go_percent"].mean(),
        "nogo": safe_df["nogo_percent"].mean(),
        "background": safe_df["background_percent"].mean(),
    }

    bdi_means = {
        "go": bdi_df["go_percent"].mean(),
        "nogo": bdi_df["nogo_percent"].mean(),
        "background": bdi_df["background_percent"].mean(),
    }

    # Visualization 1: Go zone vs No-Go zone + Background
    fig, ax = plt.subplots(figsize=(10, 6))
    x = np.arange(len(["Safe", "BDI"]))
    width = 0.35

    ax.bar(
        x - width / 2,
        [safe_means["go"], bdi_means["go"]],
        width,
        label="Go Zone",
        color="green",
        alpha=0.8,
    )
    ax.bar(
        x + width / 2,
        [
            safe_means["nogo"] + safe_means["background"],
            bdi_means["nogo"] + bdi_means["background"],
        ],
        width,
        label="No-Go + Background",
        color="red",
        alpha=0.8,
    )

    ax.set_ylabel("Percentage of TTI Pixels (%)")
    ax.set_title("TTI Distribution: Go Zone vs No-Go+Background")
    ax.set_xticks(x)
    ax.set_xticklabels(["Safe", "BDI"])
    ax.legend()
    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "viz1_go_vs_nogo_bg.png"), dpi=300)
    plt.close()

    # Visualization 2: Go zone only (simpler comparison)
    fig, ax = plt.subplots(figsize=(10, 6))

    ax.bar(
        ["Safe", "BDI"],
        [safe_means["go"], bdi_means["go"]],
        color=["green", "red"],
        alpha=0.8,
    )
    ax.set_ylabel("Percentage of TTI in Go Zone (%)")
    ax.set_title("TTI in Go Zone: Safe vs BDI Videos")
    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "viz2_go_zone_comparison.png"), dpi=300)
    plt.close()

    # Visualization 3: Stacked bar chart showing all zones
    fig, ax = plt.subplots(figsize=(10, 6))

    safe_values = [safe_means["go"], safe_means["nogo"], safe_means["background"]]
    bdi_values = [bdi_means["go"], bdi_means["nogo"], bdi_means["background"]]

    x = np.arange(2)
    width = 0.5

    ax.bar(
        x,
        [safe_values[0], bdi_values[0]],
        width,
        label="Go Zone",
        color="green",
        alpha=0.8,
    )
    ax.bar(
        x,
        [safe_values[1], bdi_values[1]],
        width,
        bottom=[safe_values[0], bdi_values[0]],
        label="No-Go Zone",
        color="red",
        alpha=0.8,
    )
    ax.bar(
        x,
        [safe_values[2], bdi_values[2]],
        width,
        bottom=[safe_values[0] + safe_values[1], bdi_values[0] + bdi_values[1]],
        label="Background",
        color="gray",
        alpha=0.8,
    )

    ax.set_ylabel("Percentage of TTI Pixels (%)")
    ax.set_title("TTI Distribution Across All Zones")
    ax.set_xticks(x)
    ax.set_xticklabels(["Safe", "BDI"])
    ax.legend()
    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "viz3_stacked_all_zones.png"), dpi=300)
    plt.close()

    # Visualization 4: Individual video scatter plot
    fig, ax = plt.subplots(figsize=(12, 6))

    ax.scatter(
        range(len(safe_df)),
        safe_df["go_percent"],
        color="green",
        marker="o",
        s=100,
        alpha=0.6,
        label="Safe - Go Zone",
    )
    ax.scatter(
        range(len(safe_df)),
        safe_df["nogo_percent"],
        color="lightcoral",
        marker="o",
        s=100,
        alpha=0.6,
        label="Safe - No-Go Zone",
    )

    offset = len(safe_df)
    ax.scatter(
        range(offset, offset + len(bdi_df)),
        bdi_df["go_percent"],
        color="darkgreen",
        marker="s",
        s=100,
        alpha=0.6,
        label="BDI - Go Zone",
    )
    ax.scatter(
        range(offset, offset + len(bdi_df)),
        bdi_df["nogo_percent"],
        color="red",
        marker="s",
        s=100,
        alpha=0.6,
        label="BDI - No-Go Zone",
    )

    ax.axvline(x=offset - 0.5, color="black", linestyle="--", linewidth=2)
    ax.set_xlabel("Video Index")
    ax.set_ylabel("Percentage of TTI Pixels (%)")
    ax.set_title("Per-Video TTI Distribution in Go and No-Go Zones")
    ax.legend()
    ax.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "viz4_per_video_scatter.png"), dpi=300)
    plt.close()

    print(f"\n✓ Saved 4 visualization plots to: {output_dir}")


def perform_statistical_analysis(combined_df: pd.DataFrame, output_dir: str):
    """
    Perform statistical comparison between Safe and BDI videos.

    Tests whether there are significant differences in:
    - Go zone percentage
    - No-Go zone percentage
    - Background percentage

    Args:
        combined_df: DataFrame with all video statistics
        output_dir: Directory to save statistical results
    """
    safe_df = combined_df[combined_df["video_type"] == "safe"]
    bdi_df = combined_df[combined_df["video_type"] == "BDI"]

    results = []

    # Test each metric
    for metric in ["go_percent", "nogo_percent", "background_percent"]:
        safe_values = safe_df[metric].values
        bdi_values = bdi_df[metric].values

        # Perform statistical test
        if USE_TTEST:
            stat, p_value = stats.ttest_ind(safe_values, bdi_values)
            test_name = "Independent t-test"
        else:
            stat, p_value = stats.mannwhitneyu(
                safe_values, bdi_values, alternative="two-sided"
            )
            test_name = "Mann-Whitney U test"

        # Calculate descriptive statistics
        safe_mean = safe_values.mean()
        safe_std = safe_values.std()
        bdi_mean = bdi_values.mean()
        bdi_std = bdi_values.std()

        significant = "Yes" if p_value < SIGNIFICANCE_LEVEL else "No"

        results.append(
            {
                "metric": metric,
                "test": test_name,
                "safe_mean": round(safe_mean, 4),
                "safe_std": round(safe_std, 4),
                "bdi_mean": round(bdi_mean, 4),
                "bdi_std": round(bdi_std, 4),
                "statistic": round(stat, 4),
                "p_value": round(p_value, 6),
                "significant": significant,
            }
        )

    # Save results
    results_df = pd.DataFrame(results)
    results_csv = os.path.join(output_dir, "statistical_comparison.csv")
    results_df.to_csv(results_csv, index=False)

    # Print summary
    print("\n" + "=" * 80)
    print("STATISTICAL COMPARISON: Safe vs BDI Videos")
    print("=" * 80)
    print(f"Test used: {test_name}")
    print(f"Significance level: {SIGNIFICANCE_LEVEL}")
    print(f"\nSafe videos: n={len(safe_df)}")
    print(f"BDI videos: n={len(bdi_df)}")
    print("\nResults:")
    print(results_df.to_string(index=False))
    print(f"\n✓ Saved statistical results to: {results_csv}")
    print("=" * 80)

    return results_df


# =============================================================================
# MAIN EXECUTION
# =============================================================================


def process_single_video_pair(args):
    """
    Wrapper function to process a single video pair (for multiprocessing).

    Args:
        args: Tuple of (idx, total, entry) where entry is the video pair configuration

    Returns:
        Dict with processing results or None if failed
    """
    idx, total, entry = args

    if not isinstance(entry, (list, tuple)) or len(entry) < 4:
        print(f"⚠ Warning: Invalid entry format at index {idx}: {entry}")
        return None

    tti_path, zone_path, sync_offset, video_type, case_id = entry[:5]

    print(f"\n[{idx}/{total}] Processing: {case_id} ({video_type})")
    print(f"  TTI video:  {os.path.basename(tti_path)}")
    print(f"  Zone video: {os.path.basename(zone_path)}")
    print(f"  Sync offset: {sync_offset}s")

    try:
        overlay_png, csv_row, video_out_dir, video_stats = compute_overlap_stats(
            tti_video_path=tti_path,
            zonemap_video_path=zone_path,
            out_dir=BASE_OUTPUT_DIR,
            video_type=video_type,
            case_id=case_id,
            interval_seconds=INTERVAL_SECONDS,
            sync_offset_s=sync_offset,
        )

        # Read summary row
        row_df = pd.read_csv(csv_row)
        summary_row = row_df.iloc[0].to_dict()

        # Create ZIP files
        images_zip, csvs_zip = None, None
        if GENERATE_ZIPS:
            images_zip, csvs_zip = make_video_zips(video_out_dir)

        print(f"  ✓ Overlay example: {overlay_png}")
        print(f"  ✓ Summary CSV:     {csv_row}")
        if GENERATE_ZIPS and images_zip:
            print(f"  ✓ Images ZIP:      {images_zip}")
        if GENERATE_ZIPS and csvs_zip:
            print(f"  ✓ CSVs ZIP:        {csvs_zip}")

        return {
            "success": True,
            "video_stats": video_stats,
            "summary_row": summary_row,
            "zip_info": {
                "video": os.path.basename(tti_path),
                "video_type": video_type,
                "case_id": case_id,
                "output_folder": video_out_dir,
                "images_zip": images_zip if images_zip else "",
                "csvs_zip": csvs_zip if csvs_zip else "",
            },
        }

    except Exception as e:
        print(f"  ✗ Failed: {e}")
        import traceback

        traceback.print_exc()
        return None


def main():
    """
    Main execution function: Process all video pairs and generate outputs.

    Steps:
    1. Process each video pair to compute overlap statistics
    2. Combine results from all videos
    3. Generate comparison visualizations
    4. Perform statistical analysis
    5. Create ZIP files for easy sharing
    """
    os.makedirs(BASE_OUTPUT_DIR, exist_ok=True)

    combined_rows = []
    zip_index_rows = []
    video_stats_list = []

    print("\n" + "=" * 80)
    print("TTI & GNG OVERLAP ANALYSIS - PROJECT ii (PARALLELIZED)")
    print("=" * 80)
    print(f"Processing {len(VIDEO_PAIRS)} video pairs...")
    print(f"Using {NUM_CPUS} CPU cores for parallel processing")
    print(f"Output directory: {BASE_OUTPUT_DIR}")
    print("=" * 80 + "\n")

    # Prepare arguments for parallel processing
    process_args = [
        (idx + 1, len(VIDEO_PAIRS), entry) for idx, entry in enumerate(VIDEO_PAIRS)
    ]

    # Process video pairs in parallel using multiprocessing Pool
    with Pool(processes=NUM_CPUS) as pool:
        results = pool.map(process_single_video_pair, process_args)

    # Collect results from parallel processing
    for result in results:
        if result is not None and result.get("success"):
            video_stats_list.append(result["video_stats"])
            combined_rows.append(result["summary_row"])
            if GENERATE_ZIPS and result["zip_info"]:
                zip_index_rows.append(result["zip_info"])

    # Generate combined outputs
    print("\n" + "=" * 80)
    print("GENERATING COMBINED OUTPUTS")
    print("=" * 80)

    if not combined_rows:
        print("⚠ No videos were successfully processed. Exiting.")
        return

    # Save combined summary CSV
    combined_csv = os.path.join(BASE_OUTPUT_DIR, "trained_all_frames_summary.csv")
    combined_df = pd.DataFrame(combined_rows)
    combined_df.to_csv(combined_csv, index=False)
    print(f"\n✓ Combined summary CSV: {combined_csv}")

    # Save ZIP index
    if zip_index_rows:
        zip_index_csv = os.path.join(BASE_OUTPUT_DIR, "trained_zip_index.csv")
        pd.DataFrame(zip_index_rows).to_csv(zip_index_csv, index=False)
        print(f"✓ ZIP index CSV: {zip_index_csv}")

    # Generate comparison visualizations
    viz_dir = os.path.join(BASE_OUTPUT_DIR, "visualizations")
    generate_comparison_visualizations(combined_df, viz_dir)

    # Perform statistical analysis
    stats_results = perform_statistical_analysis(combined_df, BASE_OUTPUT_DIR)

    # Print final summary
    print("\n" + "=" * 80)
    print("FINAL SUMMARY")
    print("=" * 80)

    safe_count = len(combined_df[combined_df["video_type"] == "safe"])
    bdi_count = len(combined_df[combined_df["video_type"] == "BDI"])

    print(f"Videos processed: {len(combined_df)} total")
    print(f"  - Safe videos: {safe_count}")
    print(f"  - BDI videos: {bdi_count}")

    # Calculate overall means
    safe_df = combined_df[combined_df["video_type"] == "safe"]
    bdi_df = combined_df[combined_df["video_type"] == "BDI"]

    if not safe_df.empty:
        print("\nSafe videos - Average TTI distribution:")
        print(f"  Go zone:     {safe_df['go_percent'].mean():.2f}%")
        print(f"  No-Go zone:  {safe_df['nogo_percent'].mean():.2f}%")
        print(f"  Background:  {safe_df['background_percent'].mean():.2f}%")

    if not bdi_df.empty:
        print("\nBDI videos - Average TTI distribution:")
        print(f"  Go zone:     {bdi_df['go_percent'].mean():.2f}%")
        print(f"  No-Go zone:  {bdi_df['nogo_percent'].mean():.2f}%")
        print(f"  Background:  {bdi_df['background_percent'].mean():.2f}%")

    print("\n" + "=" * 80)
    print("PROJECT ii ANALYSIS COMPLETE")
    print("=" * 80)
    print(f"\nAll outputs saved to: {BASE_OUTPUT_DIR}")
    print("\nDeliverables generated:")
    print("  ✓ Per-frame CSV files for each video")
    print("  ✓ Per-video summary statistics")
    print("  ✓ Combined summary CSV (all videos)")
    print("  ✓ Statistical comparison results")
    print("  ✓ 4 visualization plots (slide 8 format)")
    print("  ✓ Visualization overlays (PNG images)")
    if GENERATE_ZIPS:
        print("  ✓ ZIP files for easy sharing")
    print("\nEstimated completion date: October 3rd")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    # Required for multiprocessing on Windows and macOS
    import multiprocessing

    multiprocessing.set_start_method("spawn", force=True)
    main()
