#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Project IV — Pixel-Level Validation with PR Curve
Author: Lorenz

What it does:
  - Reads GT JSONs (Labelbox-style) and TTI inference JSONs (uses tti_bounding_box)
  - Builds per-frame binary masks on a normalized canvas from normalized boxes
  - Computes per-frame pixel TP/FP/FN, IoU, Dice
  - Aggregates per-video metrics
  - Builds a PR curve by sweeping IoU thresholds with frame-level TP/FP/FN
  - Saves:
      ./pixel_metrics_per_frame.csv
      ./pixel_metrics_per_video.csv
      ./pr_curve.png
      ./sample_confusion_table.png
"""

import glob
import json
import os
import re

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
from typing import Any, Dict, List, Optional, Tuple

import matplotlib.pyplot as plt

# =========================
# Configuration (edit here)
# =========================
GT_DIR = "/cluster/projects/madanigroup/lorenz/tti/ground_truths"  # Ground-truth JSONs
INF_DIR = "/cluster/projects/madanigroup/lorenz/tti/inferences"  # Inference JSONs (pred_*.json) — we use tti_bounding_box
OUT_DIR = "./"  # Default output path

# Canvas used to rasterize normalized boxes (scale-invariant for IoU)
CANVAS_W = 1000
CANVAS_H = 1000

# PR sweep thresholds on IoU
PR_TMIN = 0.00
PR_TMAX = 1.00
PR_STEPS = 21  # 0.00, 0.05, ..., 1.00


# =========================
# Utilities
# =========================
def ensure_out_dir(path: str):
    if path and not os.path.isdir(path):
        os.makedirs(path, exist_ok=True)


def norm_key(name: str) -> str:
    """Normalize filename key for matching (drop pred_ prefix, lower, strip, collapse spaces, drop extension)."""
    base = os.path.basename(name)
    stem, _ = os.path.splitext(base)
    if stem.startswith("pred_"):
        stem = stem[5:]
    return re.sub(r"\s+", " ", stem.strip().lower())


def list_gt_jsons(folder: str) -> List[str]:
    return sorted(glob.glob(os.path.join(folder, "*.json")))


def list_pred_jsons(folder: str) -> List[str]:
    return sorted(
        [
            p
            for p in glob.glob(os.path.join(folder, "*.json"))
            if os.path.basename(p).startswith("pred_")
        ]
    )


def _root_from_labelbox_export(data: Any) -> Optional[Dict[str, Any]]:
    """Handle list-root or dict-root exports; return first dict root or None."""
    if isinstance(data, dict):
        return data
    if isinstance(data, list) and data and isinstance(data[0], dict):
        return data[0]
    return None


def _labels_map_from_root(root: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """Try to reach labels map from data_units[...] or directly root['labels']."""
    labels = None
    du = root.get("data_units", {})
    if isinstance(du, dict) and du:
        du_key = next(iter(du.keys()))
        labels = du.get(du_key, {}).get("labels")
    if labels is None:
        labels = root.get("labels")
    return labels if isinstance(labels, dict) else None


def _norm_text(s: str) -> str:
    return re.sub(r"\s+", " ", (s or "").strip().lower())


def xywh_norm_to_pixels(box, W: int, H: int) -> Tuple[int, int, int, int]:
    """Convert normalized xywh in [0,1] to integer pixel box (x1,y1,x2,y2), clipped to canvas."""
    x = float(box.get("x", 0.0))
    y = float(box.get("y", 0.0))
    w = float(box.get("w", 0.0))
    h = float(box.get("h", 0.0))
    x1 = int(round(x * W))
    y1 = int(round(y * H))
    x2 = int(round((x + w) * W))
    y2 = int(round((y + h) * H))
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(W, x2), min(H, y2)
    return x1, y1, max(x1, x2), max(y1, y2)


def rasterize_boxes(boxes: List[Dict[str, float]], W: int, H: int) -> np.ndarray:
    """Rasterize a list of normalized xywh boxes into a binary mask on WxH canvas."""
    mask = np.zeros((H, W), dtype=np.uint8)
    for b in boxes:
        x1, y1, x2, y2 = xywh_norm_to_pixels(b, W, H)
        if x2 > x1 and y2 > y1:
            mask[y1:y2, x1:x2] = 1
    return mask


# =========================
# Parsing GT (use boundingBox)
# =========================
def parse_gt_boxes_per_frame(gt_json_path: str) -> Dict[int, List[Dict[str, float]]]:
    """
    Return: {frame_idx: [ {x,y,w,h}, ... ] } for GT TTI boxes.
    Uses objects with name/value indicating start_of_tti; uses 'boundingBox'.
    """
    with open(gt_json_path, "r") as f:
        data = json.load(f)
    root = _root_from_labelbox_export(data)
    if not root:
        return {}
    labels = _labels_map_from_root(root)
    if not labels:
        return {}

    per_frame = {}
    for frame_key, payload in labels.items():
        # Determine frame index
        fidx = None
        if isinstance(payload, dict):
            # Prefer payload["frame"] else key
            if "frame" in payload:
                try:
                    fidx = int(payload["frame"])
                except:
                    fidx = None
            if fidx is None:
                try:
                    fidx = int(frame_key)
                except:
                    fidx = None
        else:
            try:
                fidx = int(frame_key)
            except:
                fidx = None
        if fidx is None:
            continue

        objs = payload.get("objects", []) if isinstance(payload, dict) else []
        boxes = []
        for obj in objs:
            name = _norm_text(obj.get("name", ""))
            value = _norm_text(obj.get("value", ""))
            # We count GT TTI presence only if marked as start_of_tti
            if name in {"start of tti", "start_of_tti"} or value == "start_of_tti":
                bb = obj.get("boundingBox") or obj.get("bounding_box")
                if isinstance(bb, dict) and all(k in bb for k in ("x", "y", "w", "h")):
                    boxes.append(
                        {"x": bb["x"], "y": bb["y"], "w": bb["w"], "h": bb["h"]}
                    )
        if boxes:
            per_frame[fidx] = boxes
    return per_frame


# =========================
# Parsing Predictions (use tti_bounding_box ONLY)
# =========================
def parse_pred_boxes_per_frame(
    pred_json_path: str,
) -> Dict[int, List[Dict[str, float]]]:
    """
    Return: {frame_idx: [ {x,y,w,h}, ... ] } for predicted TTI boxes.
    Only uses 'tti_bounding_box' inside each object.
    """
    with open(pred_json_path, "r") as f:
        data = json.load(f)
    root = _root_from_labelbox_export(data)
    if not root:
        return {}
    labels = _labels_map_from_root(root)
    if not labels:
        return {}

    per_frame = {}
    for frame_key, payload in labels.items():
        # Frame index
        fidx = None
        if isinstance(payload, dict):
            if "frame" in payload:
                try:
                    fidx = int(payload["frame"])
                except:
                    fidx = None
            if fidx is None:
                try:
                    fidx = int(frame_key)
                except:
                    fidx = None
        else:
            try:
                fidx = int(frame_key)
            except:
                fidx = None
        if fidx is None:
            continue

        objs = payload.get("objects", []) if isinstance(payload, dict) else []
        boxes = []
        for obj in objs:
            tti_bb = obj.get("tti_bounding_box") or obj.get("tti_boundingBox")
            if isinstance(tti_bb, dict) and all(
                k in tti_bb for k in ("x", "y", "w", "h")
            ):
                boxes.append(
                    {
                        "x": tti_bb["x"],
                        "y": tti_bb["y"],
                        "w": tti_bb["w"],
                        "h": tti_bb["h"],
                    }
                )
        if boxes:
            per_frame[fidx] = boxes
    return per_frame


# =========================
# Metrics
# =========================
def pixel_metrics(
    gt_mask: np.ndarray, pred_mask: np.ndarray
) -> Tuple[int, int, int, int, float, float]:
    """
    Return TP, FP, FN, TN, IoU, Dice for one frame.
    If both masks empty (no positives), IoU/Dice defined as 0.0 for aggregation purposes (commonly dropped).
    """
    g = gt_mask.astype(bool)
    p = pred_mask.astype(bool)

    tp = int(np.logical_and(g, p).sum())
    fp = int(np.logical_and(~g, p).sum())
    fn = int(np.logical_and(g, ~p).sum())
    tn = int(np.logical_and(~g, ~p).sum())

    denom_iou = tp + fp + fn
    iou = (tp / denom_iou) if denom_iou > 0 else 0.0

    denom_dice = 2 * tp + fp + fn
    dice = (2 * tp / denom_dice) if denom_dice > 0 else 0.0

    return tp, fp, fn, tn, iou, dice


# =========================
# PR Curve (frame-level)
# =========================
def build_pr_curve(rows: List[Dict[str, Any]], out_path_png: str):
    """
    rows contain per-frame stats including:
      - gt_area, pred_area, iou
    Define:
      gt_pos  := gt_area > 0
      pred_pos := pred_area > 0
    For each IoU threshold τ:
      TP := count(gt_pos & pred_pos & iou >= τ)
      FP := count(~gt_pos & pred_pos)
      FN := count(gt_pos & (not pred_pos or iou < τ))
    """
    if not rows:
        return

    gt_pos = np.array([r["gt_area"] > 0 for r in rows], dtype=bool)
    pred_pos = np.array([r["pred_area"] > 0 for r in rows], dtype=bool)
    ious = np.array([r["iou"] for r in rows], dtype=float)

    thresholds = np.linspace(PR_TMIN, PR_TMAX, PR_STEPS)
    precisions, recalls, f1s = [], [], []

    for t in thresholds:
        tp = np.sum(gt_pos & pred_pos & (ious >= t))
        fp = np.sum((~gt_pos) & pred_pos)
        fn = np.sum(gt_pos & ((~pred_pos) | (ious < t)))

        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (2 * prec * rec / (prec + rec)) if (prec + rec) > 0 else 0.0

        precisions.append(prec)
        recalls.append(rec)
        f1s.append(f1)

    plt.figure(figsize=(7, 5))
    plt.plot(thresholds, precisions, label="Precision")
    plt.plot(thresholds, recalls, label="Recall")
    plt.plot(thresholds, f1s, label="F1")
    plt.xlabel("IoU threshold (τ)")
    plt.ylabel("Score")
    plt.title("Frame-level PR/F1 vs IoU Threshold")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path_png)
    plt.close()


# =========================
# Confusion table figure
# =========================
def save_sample_confusion_table(rows: List[Dict[str, Any]], out_path_png: str):
    """Pick a representative frame (median IoU among informative frames) and save a 2x2 table image."""
    if not rows:
        return
    informative = [r for r in rows if (r["gt_area"] + r["pred_area"]) > 0]
    if not informative:
        informative = rows
    # median IoU
    informative.sort(key=lambda r: r["iou"])
    sample = informative[len(informative) // 2]

    tp, fp, fn, tn = sample["tp"], sample["fp"], sample["fn"], sample["tn"]

    fig, ax = plt.subplots(figsize=(4, 3))
    ax.axis("off")
    table_data = [
        ["", "Pred TTI", "Pred No-TTI"],
        ["GT TTI", f"TP = {tp}", f"FN = {fn}"],
        ["GT No-TTI", f"FP = {fp}", f"TN = {tn}"],
    ]
    the_table = ax.table(cellText=table_data, loc="center", cellLoc="center")
    the_table.scale(1.2, 1.6)
    ax.set_title(
        f"Sample Pixel Confusion (video={sample['video']}, frame={sample['frame']}, IoU={sample['iou']:.2f})"
    )
    plt.tight_layout()
    plt.savefig(out_path_png, dpi=160)
    plt.close()


# =========================
# Main
# =========================
def main():
    ensure_out_dir(OUT_DIR)

    gt_jsons = list_gt_jsons(GT_DIR)
    pred_jsons = list_pred_jsons(INF_DIR)

    if not gt_jsons:
        print(f"[ERROR] No GT JSONs found in {GT_DIR}")
        return
    if not pred_jsons:
        print(f"[ERROR] No prediction JSONs (pred_*.json) found in {INF_DIR}")
        return

    gt_map = {norm_key(p): p for p in gt_jsons}
    pred_map = {norm_key(p): p for p in pred_jsons}
    common_keys = sorted(set(gt_map.keys()) & set(pred_map.keys()))
    if not common_keys:
        print("[ERROR] No matching GT and prediction JSON pairs; check filenames.")
        return

    all_frame_rows: List[Dict[str, Any]] = []
    per_video_metrics: List[Dict[str, Any]] = []

    for key in common_keys:
        gt_path = gt_map[key]
        pred_path = pred_map[key]
        video_name = os.path.splitext(os.path.basename(gt_path))[0]
        print(f"[Video] {video_name}")

        # ---- parse boxes per frame ----
        gt_boxes_pf = parse_gt_boxes_per_frame(gt_path)
        pred_boxes_pf = parse_pred_boxes_per_frame(pred_path)

        # frames considered = union of frames where GT or Pred has boxes
        frame_ids = sorted(set(gt_boxes_pf.keys()) | set(pred_boxes_pf.keys()))
        if not frame_ids:
            print("  (no frames with GT or Pred boxes) — skipping.")
            continue

        # accumulate per-frame metrics
        rows_this_video = []
        for fidx in frame_ids:
            gt_boxes = gt_boxes_pf.get(fidx, [])
            pred_boxes = pred_boxes_pf.get(fidx, [])

            gt_mask = rasterize_boxes(gt_boxes, CANVAS_W, CANVAS_H)
            pred_mask = rasterize_boxes(pred_boxes, CANVAS_W, CANVAS_H)

            tp, fp, fn, tn, iou, dice = pixel_metrics(gt_mask, pred_mask)

            rows_this_video.append(
                {
                    "video": video_name,
                    "frame": fidx,
                    "tp": tp,
                    "fp": fp,
                    "fn": fn,
                    "tn": tn,
                    "iou": iou,
                    "dice": dice,
                    "gt_area": int(gt_mask.sum()),
                    "pred_area": int(pred_mask.sum()),
                }
            )

        # aggregate per-video
        dfv = pd.DataFrame(rows_this_video)
        # informative frames = denominator > 0 for IoU/Dice
        denom = (dfv["tp"] + dfv["fp"] + dfv["fn"]) > 0
        if denom.any():
            iou_mean = float(dfv.loc[denom, "iou"].mean())
            iou_std = float(dfv.loc[denom, "iou"].std(ddof=0))
            dice_mean = float(dfv.loc[denom, "dice"].mean())
            dice_std = float(dfv.loc[denom, "dice"].std(ddof=0))
            frames_used = int(denom.sum())
        else:
            iou_mean = iou_std = dice_mean = dice_std = 0.0
            frames_used = 0

        per_video_metrics.append(
            {
                "video": video_name,
                "frames_used": frames_used,
                "iou_mean": iou_mean,
                "iou_std": iou_std,
                "dice_mean": dice_mean,
                "dice_std": dice_std,
            }
        )

        all_frame_rows.extend(rows_this_video)

    # ---- save per-frame CSV ----
    df_all = pd.DataFrame(all_frame_rows)
    per_frame_csv = os.path.join(OUT_DIR, "pixel_metrics_per_frame.csv")
    df_all.sort_values(["video", "frame"]).to_csv(per_frame_csv, index=False)
    print(f"[OK] Saved per-frame metrics: {per_frame_csv}")

    # ---- save per-video CSV ----
    df_vid = pd.DataFrame(per_video_metrics)
    per_video_csv = os.path.join(OUT_DIR, "pixel_metrics_per_video.csv")
    df_vid.sort_values(["video"]).to_csv(per_video_csv, index=False)
    print(f"[OK] Saved per-video metrics: {per_video_csv}")

    # ---- PR curve from frame-level stats ----
    pr_png = os.path.join(OUT_DIR, "pr_curve.png")
    build_pr_curve(all_frame_rows, pr_png)
    print(f"[OK] Saved PR curve: {pr_png}")

    # ---- sample confusion table ----
    conf_png = os.path.join(OUT_DIR, "sample_confusion_table.png")
    save_sample_confusion_table(all_frame_rows, conf_png)
    print(f"[OK] Saved sample confusion table: {conf_png}")


if __name__ == "__main__":
    main()
