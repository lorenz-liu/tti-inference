#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Project IV – Pixel-level evaluation of TTI vs Ground Truth.

Fixes in this version:
- Accepts JSON roots that are LISTS (common in your exports).
- Robust to missing keys / empty arrays.
"""

import glob
import json
import os
import re

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# =========================
# ======= CONSTANTS =======
# =========================

# Paths
INFERENCES_DIR = "/Users/niuniu/Desktop/inferences"  # folder with pred_*.json
GROUND_TRUTHS_DIR = (
    "/Users/niuniu/Desktop/ground_truths"  # folder with matching GT *.json
)
OUTPUT_DIR = "./"

# Evaluation raster (higher -> more accurate, slower)
RASTER_WIDTH = 1920
RASTER_HEIGHT = 1080

# Labels & keys used in JSON
PRED_EVENT_VALUE = "start_of_tti"
GT_EVENT_VALUE = "start_of_tti"
PRED_BBOX_KEY = "tti_bounding_box"  # in predictions
GT_BBOX_KEY = "boundingBox"  # in ground truths
LABELS_TOP_KEY = "data_units"  # both files
LABELS_FRAME_KEY = "labels"  # both files
OBJECTS_KEY = "objects"  # both files
VALUE_KEY = "value"  # both files

# Visualization
CMAP = "Blues"  # matplotlib colormap for confusion matrix heatmaps


# ==============================
# ======= UTIL FUNCTIONS =======
# ==============================


def ensure_output_dir(path: str):
    os.makedirs(path, exist_ok=True)


def coerce_record(j):
    """
    Accept JSON root that may be:
      - dict: return as-is
      - list:
          []           -> {}
          [dict]       -> that dict
          [dict, ...]  -> merge best-effort (first non-empty dict wins for top-level keys)
    """
    if isinstance(j, dict):
        return j
    if isinstance(j, list):
        if not j:
            return {}
        # If there's just one item, use it directly.
        if len(j) == 1 and isinstance(j[0], dict):
            return j[0]
        # If multiple items, pick the first dict that has LABELS_TOP_KEY
        for item in j:
            if isinstance(item, dict) and LABELS_TOP_KEY in item:
                return item
        # Fallback: first dict
        for item in j:
            if isinstance(item, dict):
                return item
        return {}
    # Unknown root
    return {}


def load_json(path: str):
    with open(path, "r") as f:
        raw = json.load(f)
    return coerce_record(raw)


def first_key(d: dict):
    for k in d.keys():
        return k
    return None


def extract_frame_objects(j: dict) -> dict:
    """
    Return {frame_index(int): [objects]} from our JSON format.
    Handles either list or dict roots (already coerced to dict).
    """
    du = j.get(LABELS_TOP_KEY, {})
    inner_key = first_key(du)
    if inner_key is None:
        return {}
    labels = du.get(inner_key, {}).get(LABELS_FRAME_KEY, {}) or {}
    frame_to_objects = {}
    for k, v in labels.items():
        try:
            fi = int(k)
        except Exception:
            continue
        objs = (v or {}).get(OBJECTS_KEY, []) or []
        # Guarantee list-of-dicts
        frame_to_objects[fi] = [o for o in objs if isinstance(o, dict)]
    return frame_to_objects


def objs_with_value(objs, wanted_value: str):
    return [o for o in objs if isinstance(o, dict) and o.get(VALUE_KEY) == wanted_value]


def rect_to_pixels(rect, W, H):
    """
    Convert normalized rect {x,y,w,h} in [0,1] to integer pixel bounds [x1:x2, y1:y2],
    clamped to raster. Returns (x1, x2, y1, y2) or None if invalid.
    """
    if not isinstance(rect, dict):
        return None
    try:
        x = float(rect.get("x", 0.0))
        y = float(rect.get("y", 0.0))
        w = float(rect.get("w", 0.0))
        h = float(rect.get("h", 0.0))
    except Exception:
        return None
    if w <= 0 or h <= 0:
        return None
    x1 = int(np.floor(x * W))
    y1 = int(np.floor(y * H))
    x2 = int(np.ceil((x + w) * W))
    y2 = int(np.ceil((y + h) * H))
    # Clamp
    x1 = max(0, min(W, x1))
    x2 = max(0, min(W, x2))
    y1 = max(0, min(H, y1))
    y2 = max(0, min(H, y2))
    if x2 <= x1 or y2 <= y1:
        return None
    return x1, x2, y1, y2


def paint_union_rects(mask: np.ndarray, rects_pixels):
    for r in rects_pixels:
        if r is None:
            continue
        x1, x2, y1, y2 = r
        mask[y1:y2, x1:x2] = True


def compute_confusion(pred_mask: np.ndarray, gt_mask: np.ndarray):
    tp = np.logical_and(pred_mask, gt_mask).sum(dtype=np.int64)
    fp = np.logical_and(pred_mask, np.logical_not(gt_mask)).sum(dtype=np.int64)
    fn = np.logical_and(np.logical_not(pred_mask), gt_mask).sum(dtype=np.int64)
    tn = np.logical_and(np.logical_not(pred_mask), np.logical_not(gt_mask)).sum(
        dtype=np.int64
    )
    return int(tp), int(fp), int(fn), int(tn)


def derive_metrics(tp, fp, fn, tn):
    eps = 1e-9
    precision = tp / (tp + fp + eps)
    recall = tp / (tp + fn + eps)
    f1 = 2 * precision * recall / (precision + recall + eps)
    iou = tp / (tp + fp + fn + eps)
    dice = 2 * tp / (2 * tp + fp + fn + eps)
    acc = (tp + tn) / (tp + fp + fn + tn + eps)
    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "iou": iou,
        "dice": dice,
        "accuracy": acc,
    }


def plot_confusion(tp, fp, fn, tn, title, outpath):
    cm = np.array([[tp, fp], [fn, tn]], dtype=np.int64)
    fig, ax = plt.subplots(figsize=(4.5, 4.0), dpi=150)
    im = ax.imshow(cm, cmap=CMAP)
    ax.set_title(title)
    ax.set_xlabel("Prediction")
    ax.set_ylabel("Ground Truth")
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(["TTI", "No TTI"])
    ax.set_yticklabels(["TTI", "No TTI"])
    for (i, j), val in np.ndenumerate(cm):
        ax.text(j, i, f"{val:,}", ha="center", va="center", fontsize=10)
    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.ax.set_ylabel("Pixels", rotation=90)
    plt.tight_layout()
    fig.savefig(outpath, bbox_inches="tight")
    plt.close(fig)


def plot_overall_summary(df, outpath):
    """
    Create a comprehensive overview figure showing:
    - Confusion matrix heatmap
    - Key metrics bar chart
    - Per-video performance comparison
    """
    # Separate global from per-video results
    global_row = (
        df[df["video"] == "__GLOBAL__"].iloc[0]
        if "__GLOBAL__" in df["video"].values
        else None
    )
    video_df = df[df["video"] != "__GLOBAL__"].copy()

    if global_row is None:
        print("[WARN] No global row found for overall summary figure.")
        return

    fig = plt.figure(figsize=(16, 10), dpi=150)
    gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)

    # ===== 1. Global Confusion Matrix (top-left) =====
    ax1 = fig.add_subplot(gs[0, 0])
    cm = np.array(
        [[global_row["tp"], global_row["fp"]], [global_row["fn"], global_row["tn"]]],
        dtype=np.int64,
    )
    im1 = ax1.imshow(cm, cmap=CMAP, aspect="auto")
    ax1.set_title(
        "Global Confusion Matrix\n(All Videos Combined)", fontsize=11, fontweight="bold"
    )
    ax1.set_xlabel("Prediction")
    ax1.set_ylabel("Ground Truth")
    ax1.set_xticks([0, 1])
    ax1.set_yticks([0, 1])
    ax1.set_xticklabels(["TTI", "No TTI"])
    ax1.set_yticklabels(["TTI", "No TTI"])
    for (i, j), val in np.ndenumerate(cm):
        ax1.text(j, i, f"{val:,}", ha="center", va="center", fontsize=9)
    plt.colorbar(im1, ax=ax1, fraction=0.046, pad=0.04)

    # ===== 2. Global Metrics Bar Chart (top-middle & top-right) =====
    ax2 = fig.add_subplot(gs[0, 1:])
    metrics_names = ["Precision", "Recall", "F1", "IoU", "Dice", "Accuracy"]
    metrics_values = [
        global_row["precision"],
        global_row["recall"],
        global_row["f1"],
        global_row["iou"],
        global_row["dice"],
        global_row["accuracy"],
    ]
    colors = plt.cm.viridis(np.linspace(0.3, 0.9, len(metrics_names)))
    bars = ax2.barh(
        metrics_names, metrics_values, color=colors, edgecolor="black", linewidth=0.8
    )
    ax2.set_xlim(0, 1.0)
    ax2.set_xlabel("Score", fontsize=10)
    ax2.set_title("Global Performance Metrics", fontsize=11, fontweight="bold")
    ax2.grid(axis="x", alpha=0.3, linestyle="--")
    for i, (bar, val) in enumerate(zip(bars, metrics_values)):
        ax2.text(val + 0.02, i, f"{val:.3f}", va="center", fontsize=9)

    # ===== 3. Per-Video F1 Scores (middle row) =====
    if len(video_df) > 0:
        ax3 = fig.add_subplot(gs[1, :])
        video_df_sorted = video_df.sort_values("f1", ascending=True)
        video_names = video_df_sorted["video"].tolist()
        f1_scores = video_df_sorted["f1"].tolist()

        bar_colors = [
            "#d62728" if f1 < 0.5 else "#ff7f0e" if f1 < 0.8 else "#2ca02c"
            for f1 in f1_scores
        ]
        bars3 = ax3.barh(
            video_names, f1_scores, color=bar_colors, edgecolor="black", linewidth=0.8
        )
        ax3.set_xlim(0, 1.0)
        ax3.set_xlabel("F1 Score", fontsize=10)
        ax3.set_title("Per-Video F1 Score Comparison", fontsize=11, fontweight="bold")
        ax3.grid(axis="x", alpha=0.3, linestyle="--")
        for i, (bar, val) in enumerate(zip(bars3, f1_scores)):
            ax3.text(val + 0.02, i, f"{val:.3f}", va="center", fontsize=8)
    else:
        ax3 = fig.add_subplot(gs[1, :])
        ax3.text(
            0.5,
            0.5,
            "No per-video data available",
            ha="center",
            va="center",
            fontsize=12,
        )
        ax3.set_xlim(0, 1)
        ax3.set_ylim(0, 1)
        ax3.axis("off")

    # ===== 4. Per-Video Metrics Comparison (bottom row) =====
    if len(video_df) > 0:
        ax4 = fig.add_subplot(gs[2, :])
        x = np.arange(len(video_df))
        width = 0.15

        precision_vals = video_df_sorted["precision"].tolist()
        recall_vals = video_df_sorted["recall"].tolist()
        f1_vals = video_df_sorted["f1"].tolist()
        iou_vals = video_df_sorted["iou"].tolist()

        ax4.bar(
            x - 1.5 * width,
            precision_vals,
            width,
            label="Precision",
            color="#1f77b4",
            edgecolor="black",
            linewidth=0.5,
        )
        ax4.bar(
            x - 0.5 * width,
            recall_vals,
            width,
            label="Recall",
            color="#ff7f0e",
            edgecolor="black",
            linewidth=0.5,
        )
        ax4.bar(
            x + 0.5 * width,
            f1_vals,
            width,
            label="F1",
            color="#2ca02c",
            edgecolor="black",
            linewidth=0.5,
        )
        ax4.bar(
            x + 1.5 * width,
            iou_vals,
            width,
            label="IoU",
            color="#d62728",
            edgecolor="black",
            linewidth=0.5,
        )

        ax4.set_xlabel("Video", fontsize=10)
        ax4.set_ylabel("Score", fontsize=10)
        ax4.set_title("Per-Video Metrics Comparison", fontsize=11, fontweight="bold")
        ax4.set_xticks(x)
        ax4.set_xticklabels(video_names, rotation=45, ha="right", fontsize=8)
        ax4.set_ylim(0, 1.0)
        ax4.legend(loc="upper left", fontsize=9)
        ax4.grid(axis="y", alpha=0.3, linestyle="--")
    else:
        ax4 = fig.add_subplot(gs[2, :])
        ax4.text(
            0.5,
            0.5,
            "No per-video data available",
            ha="center",
            va="center",
            fontsize=12,
        )
        ax4.set_xlim(0, 1)
        ax4.set_ylim(0, 1)
        ax4.axis("off")

    fig.suptitle(
        "Project IV: TTI Detection Evaluation - Overall Summary",
        fontsize=14,
        fontweight="bold",
        y=0.995,
    )

    fig.savefig(outpath, bbox_inches="tight", dpi=150)
    plt.close(fig)
    print(f"[OK] Overall summary figure saved: {outpath}")


def normalize_video_key(json_dict):
    """
    Build a short 'video key' from data_title if present.
    Works even if root was originally an array (already coerced).
    """
    try:
        du = json_dict.get(LABELS_TOP_KEY, {})
        inner_key = first_key(du)
        if inner_key:
            dt = du[inner_key].get("data_title")
            if isinstance(dt, str) and dt.strip():
                return dt
    except Exception:
        pass
    # fallback to dataset_title if available
    dt = json_dict.get("data_title")
    if isinstance(dt, str) and dt.strip():
        return dt.strip()
    return ""


# ======================================
# ======= MAIN EVALUATION PIPELINE =====
# ======================================


def prediction_to_gt_name(pred_filename: str) -> str:
    """
    Convert a prediction filename to the expected GT filename.
    Examples:
        pred_LapChol Case 0001 03.json  -> LapChol Case 0001 03.json
        pred_V10-Trimmed.json           -> V10 Trimmed.json
        pred_V14_Trimmed.json           -> V14 Trimmed.json
    """
    name = pred_filename
    if name.startswith("pred_"):
        name = name[len("pred_") :]
    name = re.sub(r"[-_]\s*Trimmed\.json$", " Trimmed.json", name)
    return name


def main():
    ensure_output_dir(OUTPUT_DIR)

    pred_files = sorted(glob.glob(os.path.join(INFERENCES_DIR, "*.json")))
    if not pred_files:
        print(f"[WARN] No prediction JSONs found under {INFERENCES_DIR}/")
        return

    rows = []
    global_tp = global_fp = global_fn = global_tn = 0

    gt_index = {
        os.path.basename(p): p
        for p in glob.glob(os.path.join(GROUND_TRUTHS_DIR, "*.json"))
    }

    for pred_path in pred_files:
        pred_name = os.path.basename(pred_path)
        gt_name = prediction_to_gt_name(pred_name)
        gt_path = gt_index.get(gt_name)

        if gt_path is None:
            print(
                f"[SKIP] No GT file for prediction: {pred_name}  -> expected '{gt_name}'"
            )
            continue

        # Load (and coerce) JSONs
        try:
            pred_json = load_json(pred_path)
            gt_json = load_json(gt_path)
        except Exception as e:
            print(f"[ERROR] Loading JSON failed for {pred_name} or {gt_name}: {e}")
            continue

        pred_frames = extract_frame_objects(pred_json)
        gt_frames = extract_frame_objects(gt_json)

        video_key = normalize_video_key(pred_json) or gt_name.replace(".json", "")

        tp = fp = fn = tn = 0
        evaluated_frames = 0

        matched_frames = set(pred_frames.keys()) & set(gt_frames.keys())
        for fidx in sorted(matched_frames):
            pred_objs_all = pred_frames.get(fidx, [])
            gt_objs_all = gt_frames.get(fidx, [])

            # Only frames where BOTH have at least one "start_of_tti"
            pred_tti_objs = objs_with_value(pred_objs_all, PRED_EVENT_VALUE)
            gt_tti_objs = objs_with_value(gt_objs_all, GT_EVENT_VALUE)
            if not pred_tti_objs or not gt_tti_objs:
                continue

            pred_rects = []
            for o in pred_tti_objs:
                rect = o.get(PRED_BBOX_KEY)
                pred_rects.append(rect_to_pixels(rect, RASTER_WIDTH, RASTER_HEIGHT))

            gt_rects = []
            for o in gt_tti_objs:
                rect = o.get(GT_BBOX_KEY)
                gt_rects.append(rect_to_pixels(rect, RASTER_WIDTH, RASTER_HEIGHT))

            pred_mask = np.zeros((RASTER_HEIGHT, RASTER_WIDTH), dtype=bool)
            gt_mask = np.zeros((RASTER_HEIGHT, RASTER_WIDTH), dtype=bool)
            paint_union_rects(pred_mask, pred_rects)
            paint_union_rects(gt_mask, gt_rects)

            _tp, _fp, _fn, _tn = compute_confusion(pred_mask, gt_mask)
            tp += _tp
            fp += _fp
            fn += _fn
            tn += _tn
            evaluated_frames += 1

        vid_metrics = derive_metrics(tp, fp, fn, tn)

        # Save per-video matrix
        safe_key = re.sub(r"[^A-Za-z0-9_\-]+", "_", video_key) or "unknown_video"
        cm_path = os.path.join(OUTPUT_DIR, f"confusion_matrix_{safe_key}.png")
        plot_confusion(tp, fp, fn, tn, f"{video_key} — Pixel Confusion", cm_path)

        rows.append(
            {
                "video": video_key,
                "pred_file": pred_name,
                "gt_file": gt_name,
                "evaluated_frames": evaluated_frames,
                "tp": tp,
                "fp": fp,
                "fn": fn,
                "tn": tn,
                **vid_metrics,
            }
        )

        global_tp += tp
        global_fp += fp
        global_fn += fn
        global_tn += tn

    # Global
    global_metrics = derive_metrics(global_tp, global_fp, global_fn, global_tn)
    global_cm_path = os.path.join(OUTPUT_DIR, "confusion_matrix_GLOBAL.png")
    plot_confusion(
        global_tp,
        global_fp,
        global_fn,
        global_tn,
        "GLOBAL — Pixel Confusion",
        global_cm_path,
    )

    rows.append(
        {
            "video": "__GLOBAL__",
            "pred_file": "",
            "gt_file": "",
            "evaluated_frames": sum(r.get("evaluated_frames", 0) for r in rows),
            "tp": global_tp,
            "fp": global_fp,
            "fn": global_fn,
            "tn": global_tn,
            **global_metrics,
        }
    )

    df = pd.DataFrame(rows)
    csv_path = os.path.join(OUTPUT_DIR, "project_iv_summary.csv")
    ensure_output_dir(OUTPUT_DIR)
    df.to_csv(csv_path, index=False)

    # Generate overall summary figure
    overall_fig_path = os.path.join(OUTPUT_DIR, "overall_summary.png")
    plot_overall_summary(df, overall_fig_path)

    print(f"\n[OK] Wrote CSV: {csv_path}")
    print(f"[OK] Wrote plots to: {OUTPUT_DIR}/")
    print("\nColumns:\n", ", ".join(df.columns))


if __name__ == "__main__":
    main()
