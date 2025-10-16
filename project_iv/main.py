#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Project IV — Event-Only Scoring (NO ranges)

You asked to ONLY evaluate frames where the prediction explicitly labels:
  • start_of_tti        (positive)  → score ONLY if GT also has Start of TTI at the SAME frame  → counts as TP row
  • start_of_no_interaction (negative) → score ONLY if GT also has End of TTI_ at the SAME frame → counts as TN row

For each scored frame we compute pixel confusion (TP/FP/FN/TN) + IoU + DICE
using:
  • Prediction mask  : union of all `tti_bounding_box` at that frame
  • Ground-truth mask: union of all `boundingBox` at that frame

We skip all other frames (no matching GT event).
Outputs:
  ./event_only_per_frame.csv
  ./event_only_per_video.csv
"""

import argparse
import glob
import json
import os
import re
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

# =============================================================================
# ============================== CONFIG / CONSTANTS ============================
# =============================================================================

# ---- I/O PATHS (defaults) ----
DEFAULT_GT_DIR = "/Users/niuniu/Desktop/ground_truths"
DEFAULT_PRED_DIR = "/Users/niuniu/Desktop/inferences"  # contains pred_*.json
DEFAULT_OUT_DIR = "./"

# ---- CANVAS SIZE FOR RASTERIZATION (scale-invariant; boxes are normalized) ----
CANVAS_W = 1000
CANVAS_H = 1000

# ---- LABELS (case/space tolerant; normalized internally) ----
# Ground truth events we are allowed to match on:
GT_START_LABELS = {"start of tti", "start_of_tti"}  # positive GT event
GT_END_LABELS = {"end of tti", "end_of_tti", "end_of_tti_"}  # negative GT event

# Prediction events we will evaluate:
PRED_POS_LABELS = {"start of tti", "start_of_tti"}  # positive Pred event
PRED_NEG_LABELS = {
    "start of no interaction",
    "start_of_no_interaction",
    "start_of_no_interaction_",
}  # negative Pred event

# Output files
PER_FRAME_CSV = "event_only_per_frame.csv"
PER_VIDEO_CSV = "event_only_per_video.csv"

# =============================================================================
# ============================== UTILS / HELPERS ==============================
# =============================================================================


def ensure_dir(p: str):
    if p and not os.path.isdir(p):
        os.makedirs(p, exist_ok=True)


def _norm_text(s: str) -> str:
    return re.sub(r"\s+", " ", (s or "").strip().lower())


def load_json_any(path: str) -> Optional[Dict[str, Any]]:
    """Support list-root or dict-root Labelbox exports and return the first dict-like root."""
    with open(path, "r") as f:
        data = json.load(f)
    if isinstance(data, dict):
        return data
    if isinstance(data, list) and data and isinstance(data[0], dict):
        return data[0]
    return None


def get_labels_map(root: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """Return labels map from data_units[...] or root['labels']."""
    labels = None
    du = root.get("data_units", {})
    if isinstance(du, dict) and du:
        du_key = next(iter(du.keys()))
        labels = du.get(du_key, {}).get("labels")
    if labels is None:
        labels = root.get("labels")
    return labels if isinstance(labels, dict) else None


def list_jsons(folder: str, prefix: Optional[str] = None) -> List[str]:
    paths = sorted(glob.glob(os.path.join(folder, "*.json")))
    if prefix:
        paths = [p for p in paths if os.path.basename(p).startswith(prefix)]
    return paths


def norm_key(p: str) -> str:
    base = os.path.basename(p)
    stem, _ = os.path.splitext(base)
    if stem.startswith("pred_"):
        stem = stem[5:]
    return _norm_text(stem)


# =============================================================================
# ============================ EVENT → FRAME MAPS =============================
# =============================================================================


def collect_frame_events(
    labels: Dict[str, Any], use_pred_boxes: bool
) -> Dict[int, List[Tuple[str, Dict[str, float]]]]:
    """
    Build a map: frame_idx -> list of (event_name_norm, box_dict)
    For GT we read 'boundingBox'; for Pred we read 'tti_bounding_box'.
    """
    out: Dict[int, List[Tuple[str, Dict[str, float]]]] = {}
    for k, payload in labels.items():
        if isinstance(payload, dict) and "frame" in payload:
            try:
                fidx = int(payload["frame"])
            except Exception:
                try:
                    fidx = int(k)
                except Exception:
                    continue
        else:
            try:
                fidx = int(k)
            except Exception:
                continue

        objs = payload.get("objects", []) if isinstance(payload, dict) else []
        for obj in objs:
            name = _norm_text(obj.get("name", ""))
            box = (
                (obj.get("tti_bounding_box") or obj.get("tti_boundingBox"))
                if use_pred_boxes
                else (obj.get("boundingBox") or obj.get("bounding_box"))
            )
            if isinstance(box, dict) and all(t in box for t in ("x", "y", "w", "h")):
                out.setdefault(fidx, []).append((name, box))
    return out


def collect_boxes_for_labels(
    events_at_frame: List[Tuple[str, Dict[str, float]]], wanted_labels: set
) -> List[Dict[str, float]]:
    return [box for (nm, box) in events_at_frame if nm in wanted_labels]


# =============================================================================
# =============================== MASKS & METRICS =============================
# =============================================================================


def rasterize_boxes_norm(boxes: List[Dict[str, float]], W: int, H: int) -> np.ndarray:
    """
    Rasterize normalized xywh boxes into a binary mask (H,W). Union of all boxes.
    """
    m = np.zeros((H, W), dtype=np.uint8)
    for b in boxes:
        x1 = int(round(float(b["x"]) * W))
        y1 = int(round(float(b["y"]) * H))
        x2 = int(round((float(b["x"]) + float(b["w"])) * W))
        y2 = int(round((float(b["y"]) + float(b["h"])) * H))
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(W, x2), min(H, y2)
        if x2 > x1 and y2 > y1:
            m[y1:y2, x1:x2] = 1
    return m


def pixel_confusion_and_scores(
    gt_mask: np.ndarray, pr_mask: np.ndarray
) -> Tuple[int, int, int, int, float, float]:
    g = gt_mask.astype(bool)
    p = pr_mask.astype(bool)
    tp = int(np.logical_and(g, p).sum())
    fp = int(np.logical_and(~g, p).sum())
    fn = int(np.logical_and(g, ~p).sum())
    tn = int(np.logical_and(~g, ~p).sum())

    denom_iou = tp + fp + fn
    iou = (tp / denom_iou) if denom_iou > 0 else 0.0
    denom_dice = 2 * tp + fp + fn
    dice = (2 * tp / denom_dice) if denom_dice > 0 else 0.0
    return tp, fp, fn, tn, iou, dice


# =============================================================================
# ================================== MAIN =====================================
# =============================================================================


def main():
    ap = argparse.ArgumentParser("Event-only scoring for exact frames (no ranges)")
    ap.add_argument(
        "--gt_dir", default=DEFAULT_GT_DIR, help="Ground truth JSON directory"
    )
    ap.add_argument(
        "--pred_dir",
        default=DEFAULT_PRED_DIR,
        help="Prediction JSON directory (pred_*.json)",
    )
    ap.add_argument(
        "--out_dir", default=DEFAULT_OUT_DIR, help="Output directory (default: ./)"
    )
    ap.add_argument("--canvas_w", type=int, default=CANVAS_W)
    ap.add_argument("--canvas_h", type=int, default=CANVAS_H)
    args = ap.parse_args()

    ensure_dir(args.out_dir)

    gt_files = list_jsons(args.gt_dir)
    pred_files = list_jsons(args.pred_dir, prefix="pred_")

    if not gt_files:
        print(f"[ERROR] No GT JSONs found in {args.gt_dir}")
        return
    if not pred_files:
        print(f"[ERROR] No Pred JSONs (pred_*.json) found in {args.pred_dir}")
        return

    gt_map = {norm_key(p): p for p in gt_files}
    pred_map = {norm_key(p): p for p in pred_files}
    common = sorted(set(gt_map.keys()) & set(pred_map.keys()))
    if not common:
        print("[ERROR] No matching GT / Pred JSON pairs after normalization.")
        return

    per_frame_rows: List[Dict[str, Any]] = []
    per_video_rows: List[Dict[str, Any]] = []

    for key in common:
        gt_path = gt_map[key]
        pred_path = pred_map[key]
        video_name = os.path.splitext(os.path.basename(gt_path))[0]
        print(f"[PAIR] {video_name}")

        gt_root = load_json_any(gt_path)
        pred_root = load_json_any(pred_path)
        if not gt_root or not pred_root:
            print("  [skip] invalid JSON format")
            continue

        gt_labels = get_labels_map(gt_root)
        pred_labels = get_labels_map(pred_root)
        if not gt_labels or not pred_labels:
            print("  [skip] labels missing")
            continue

        # Build frame → events maps
        gt_events = collect_frame_events(gt_labels, use_pred_boxes=False)
        pred_events = collect_frame_events(pred_labels, use_pred_boxes=True)

        # Consider ONLY frames where prediction has start_of_tti OR start_of_no_interaction
        pred_frames = sorted(pred_events.keys())

        rows_this_video: List[Dict[str, Any]] = []
        for f in pred_frames:
            ev_pred = pred_events.get(f, [])
            if not ev_pred:
                continue

            # Prediction-positive (start_of_tti) frames — score ONLY if GT has Start-of-TTI at same frame
            pred_pos_boxes = collect_boxes_for_labels(ev_pred, PRED_POS_LABELS)
            if pred_pos_boxes:
                gt_ev = gt_events.get(f, [])
                gt_pos_boxes = collect_boxes_for_labels(gt_ev, GT_START_LABELS)
                if gt_pos_boxes:
                    # Build masks as unions; compute metrics (this row counts as TP case)
                    pr_mask = rasterize_boxes_norm(
                        pred_pos_boxes, args.canvas_w, args.canvas_h
                    )
                    gt_mask = rasterize_boxes_norm(
                        gt_pos_boxes, args.canvas_w, args.canvas_h
                    )
                    tp, fp, fn, tn, iou, dice = pixel_confusion_and_scores(
                        gt_mask, pr_mask
                    )
                    rows_this_video.append(
                        {
                            "video": video_name,
                            "frame": f,
                            "case": "TP",  # per your definition
                            "tp": tp,
                            "fp": fp,
                            "fn": fn,
                            "tn": tn,
                            "iou": iou,
                            "dice": dice,
                            "gt_area": int(gt_mask.sum()),
                            "pred_area": int(pr_mask.sum()),
                        }
                    )
                # else: no matching GT start at this frame → skip completely

            # Prediction-negative (start_of_no_interaction) frames — score ONLY if GT has End-of-TTI_ at same frame
            pred_neg_boxes = collect_boxes_for_labels(ev_pred, PRED_NEG_LABELS)
            if pred_neg_boxes:
                gt_ev = gt_events.get(f, [])
                gt_neg_boxes = collect_boxes_for_labels(gt_ev, GT_END_LABELS)
                if gt_neg_boxes:
                    # Build masks; compute metrics (this row counts as TN case)
                    pr_mask = rasterize_boxes_norm(
                        pred_neg_boxes, args.canvas_w, args.canvas_h
                    )
                    gt_mask = rasterize_boxes_norm(
                        gt_neg_boxes, args.canvas_w, args.canvas_h
                    )
                    tp, fp, fn, tn, iou, dice = pixel_confusion_and_scores(
                        gt_mask, pr_mask
                    )
                    rows_this_video.append(
                        {
                            "video": video_name,
                            "frame": f,
                            "case": "TN",  # per your definition
                            "tp": tp,
                            "fp": fp,
                            "fn": fn,
                            "tn": tn,
                            "iou": iou,
                            "dice": dice,
                            "gt_area": int(gt_mask.sum()),
                            "pred_area": int(pr_mask.sum()),
                        }
                    )
                # else: no matching GT end at this frame → skip completely

        # Summarize this video
        if rows_this_video:
            dfv = pd.DataFrame(rows_this_video)
            informative = (dfv["tp"] + dfv["fp"] + dfv["fn"]) > 0
            if informative.any():
                iou_mean = float(dfv.loc[informative, "iou"].mean())
                dice_mean = float(dfv.loc[informative, "dice"].mean())
            else:
                iou_mean = dice_mean = 0.0

            per_video_rows.append(
                {
                    "video": video_name,
                    "scored_frames": len(rows_this_video),
                    "iou_mean": iou_mean,
                    "dice_mean": dice_mean,
                    "tp_rows": int((dfv["case"] == "TP").sum()),
                    "tn_rows": int((dfv["case"] == "TN").sum()),
                }
            )

        per_frame_rows.extend(rows_this_video)

    # Save results
    df_all = pd.DataFrame(per_frame_rows).sort_values(["video", "frame"])
    out_frame_csv = os.path.join(args.out_dir, PER_FRAME_CSV)
    df_all.to_csv(out_frame_csv, index=False)
    print(f"[OK] {out_frame_csv}")

    df_vid = pd.DataFrame(per_video_rows).sort_values(["video"])
    out_video_csv = os.path.join(args.out_dir, PER_VIDEO_CSV)
    df_vid.to_csv(out_video_csv, index=False)
    print(f"[OK] {out_video_csv}")

    print("\n✅ Done. (Scored only exact matching-event frames as requested.)")


if __name__ == "__main__":
    main()
