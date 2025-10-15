#!/usr/bin/env python3
# ==========================================================
# Project II: TTI & GNG Overlap Analysis (Video-to-Video)
# Reads pairs of videos:
#   - GNG (no-background) zone masks (Go = green, No-Go = red)
#   - TTI predictions (tool–tissue contact mask/video)
# Computes per-frame overlaps and aggregates per video.
# Output: CSV with mean %TTI in Go and No-Go per video.
# ----------------------------------------------------------
# Assumptions:
# - GNG videos live in:  /cluster/projects/madanigroup/lorenz/tti/gng_predictions_no_background
# - TTI videos live in:  /cluster/projects/madanigroup/lorenz/tti/inferences
# - TTI filenames are prefixed with 'pred_' and otherwise match GNG names.
# - Background in GNG is black. Go is green, No-Go is red.
# - TTI frames are masks/overlays where TTI pixels are non-black.
# You can tweak HSV ranges below if your colors differ slightly.
# ==========================================================

import csv
import glob
import os

import cv2
import numpy as np

# ---------------------------
# CONFIG: paths & parameters
# ---------------------------
GNG_DIR = "/cluster/projects/madanigroup/lorenz/tti/gng_predictions_no_background"
TTI_DIR = "/cluster/projects/madanigroup/lorenz/tti/inferences"
OUTPUT_CSV = "/cluster/projects/madanigroup/lorenz/tti/project_ii_overlap_video.csv"

# Frame sampling (1 = use every frame; increase to speed up)
FRAME_STEP = 1

# HSV thresholds (tune if needed)
GO_HSV_LO = np.array([40, 40, 40], dtype=np.uint8)  # green (Go)
GO_HSV_HI = np.array([85, 255, 255], dtype=np.uint8)

RED1_LO = np.array([0, 70, 50], dtype=np.uint8)  # red (No-Go) low hue
RED1_HI = np.array([10, 255, 255], dtype=np.uint8)
RED2_LO = np.array([170, 70, 50], dtype=np.uint8)  # red (No-Go) high hue
RED2_HI = np.array([180, 255, 255], dtype=np.uint8)


# ---------------------------
# Utilities
# ---------------------------
def norm_name_for_match(path):
    """Return a normalized base name for matching between pred_*.mp4 and *.MP4"""
    base = os.path.basename(path)
    name, ext = os.path.splitext(base)
    # Strip 'pred_' if present
    if name.startswith("pred_"):
        name = name[len("pred_") :]
    # Normalize whitespace & case for matching
    return name.strip().lower()


def list_videos(folder):
    exts = ("*.mp4", "*.MP4", "*.mov", "*.MOV")
    files = []
    for e in exts:
        files.extend(glob.glob(os.path.join(folder, e)))
    return sorted(files)


def tti_mask_from_frame(frame_bgr):
    """Return a binary mask where TTI pixels are non-black."""
    if frame_bgr is None:
        return None
    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    # Any non-zero intensity counts as TTI
    _, mask = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY)
    return mask


def zones_from_gng(frame_bgr):
    """Extract Go and No-Go binary masks from GNG BGR frame using HSV."""
    if frame_bgr is None:
        return None, None
    hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)
    go = cv2.inRange(hsv, GO_HSV_LO, GO_HSV_HI)
    red1 = cv2.inRange(hsv, RED1_LO, RED1_HI)
    red2 = cv2.inRange(hsv, RED2_LO, RED2_HI)
    nogo = cv2.bitwise_or(red1, red2)
    return go, nogo


def process_pair(gng_path, tti_path, frame_step=1):
    cap_gng = cv2.VideoCapture(gng_path)
    cap_tti = cv2.VideoCapture(tti_path)

    n_gng = int(cap_gng.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    n_tti = int(cap_tti.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    n = max(0, min(n_gng, n_tti))

    if n == 0:
        cap_gng.release()
        cap_tti.release()
        return dict(
            mean_go=np.nan, mean_nogo=np.nan, frames_considered=0, frames_with_tti=0
        )

    go_ratios = []
    nogo_ratios = []
    frames_with_tti = 0
    frames_considered = 0

    for idx in range(0, n, frame_step):
        # Seek & read both streams at same index
        cap_gng.set(cv2.CAP_PROP_POS_FRAMES, idx)
        cap_tti.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ok_gng, gng_frame = cap_gng.read()
        ok_tti, tti_frame = cap_tti.read()
        if not (ok_gng and ok_tti):
            continue

        # Resize TTI to GNG size if needed
        if gng_frame.shape[:2] != tti_frame.shape[:2]:
            tti_frame = cv2.resize(
                tti_frame,
                (gng_frame.shape[1], gng_frame.shape[0]),
                interpolation=cv2.INTER_NEAREST,
            )

        go_mask, nogo_mask = zones_from_gng(gng_frame)
        tti_mask = tti_mask_from_frame(tti_frame)
        if go_mask is None or tti_mask is None:
            continue

        frames_considered += 1

        tti_pix = np.count_nonzero(tti_mask)
        if tti_pix == 0:
            # No TTI in this frame; skip contribution to ratios
            continue

        frames_with_tti += 1

        go_overlap = np.count_nonzero(cv2.bitwise_and(tti_mask, go_mask))
        nogo_overlap = np.count_nonzero(cv2.bitwise_and(tti_mask, nogo_mask))

        go_ratios.append(go_overlap / tti_pix)
        nogo_ratios.append(nogo_overlap / tti_pix)

    cap_gng.release()
    cap_tti.release()

    mean_go = float(np.mean(go_ratios)) if go_ratios else 0.0
    mean_nogo = float(np.mean(nogo_ratios)) if nogo_ratios else 0.0

    return dict(
        mean_go=mean_go,
        mean_nogo=mean_nogo,
        frames_considered=frames_considered,
        frames_with_tti=frames_with_tti,
    )


# ---------------------------
# Build match list
# ---------------------------
gng_videos = list_videos(GNG_DIR)
tti_videos = [
    p for p in list_videos(TTI_DIR) if os.path.basename(p).startswith("pred_")
]

gng_map = {norm_name_for_match(p): p for p in gng_videos}
tti_map = {norm_name_for_match(p): p for p in tti_videos}

# Try to match by normalized base name
common_keys = sorted(set(gng_map.keys()) & set(tti_map.keys()))

if not common_keys:
    print("No matching GNG/TTI video pairs found. Check filenames.")
    exit(1)

# ---------------------------
# Run analysis & write CSV
# ---------------------------
os.makedirs(os.path.dirname(OUTPUT_CSV), exist_ok=True)
with open(OUTPUT_CSV, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(
        [
            "video_name",
            "gng_video",
            "tti_video",
            "frame_step",
            "mean_tti_in_go",
            "mean_tti_in_nogo",
            "frames_considered",
            "frames_with_tti",
        ]
    )

    for key in common_keys:
        gng_path = gng_map[key]
        tti_path = tti_map[key]
        video_name = os.path.splitext(os.path.basename(gng_path))[0]

        print(f"Processing: {video_name}")
        stats = process_pair(gng_path, tti_path, frame_step=FRAME_STEP)

        writer.writerow(
            [
                video_name,
                os.path.basename(gng_path),
                os.path.basename(tti_path),
                FRAME_STEP,
                f"{stats['mean_go']:.6f}",
                f"{stats['mean_nogo']:.6f}",
                stats["frames_considered"],
                stats["frames_with_tti"],
            ]
        )

print(f"✅ Done. Saved: {OUTPUT_CSV}")
