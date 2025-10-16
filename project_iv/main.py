#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Project IV — Range-based Pixel Evaluation (GT boundingBox vs Pred tti_bounding_box)

What this script does
---------------------
1) Parses GT and Prediction JSONs (Labelbox-style, list/dict root supported)
2) Converts boundary labels into inclusive frame ranges:
     • GT  : Start of TTI  → End of TTI_        (inclusive)
     • Pred: Start of TTI  → Start of No Interaction (inclusive)
3) For every frame inside each range, creates a binary mask by:
     • taking the boundary boxes at start/end
     • linear-interpolating (if both ends exist), or carrying forward (if only one end)
     • GT uses      `boundingBox`
     • Prediction uses `tti_bounding_box`  (as requested)
4) Computes pixel-level TP/FP/FN/TN, IoU, Dice per frame
5) Aggregates per-video metrics and saves:
     ./pixel_metrics_per_frame.csv
     ./pixel_metrics_per_video.csv
     ./pr_curve.png         (Precision/Recall/F1 vs IoU threshold)
     ./sample_confusion_table.png

Defaults assume your cluster layout. All constants & labels are at the top.
"""

import argparse
import glob
import json
import os
import re
from typing import Any, Dict, List, Optional, Tuple

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt

# =============================================================================
# ============================== CONFIG / CONSTANTS ============================
# =============================================================================

# ---- I/O PATHS (defaults) ----
DEFAULT_GT_DIR = "/cluster/projects/madanigroup/lorenz/tti/ground_truths"
DEFAULT_PRED_DIR = (
    "/cluster/projects/madanigroup/lorenz/tti/inferences"  # contains pred_*.json
)
DEFAULT_OUT_DIR = "./"

# ---- CANVAS SIZE FOR RASTERIZATION (scale-invariant; boxes are normalized) ----
CANVAS_W = 1000
CANVAS_H = 1000

# ---- LABELS (case/space tolerant; normalized internally) ----
# Ground truth: inclusive range Start of TTI → End of TTI_
GT_START_LABELS = {"start of tti", "start_of_tti"}
GT_END_LABELS = {"end of tti", "end_of_tti", "end_of_tti_"}

# Prediction: inclusive range Start of TTI → Start of No Interaction
PRED_START_LABELS = {"start of tti", "start_of_tti"}
PRED_STOP_LABELS = {
    "start of no interaction",
    "start_of_no_interaction",
    "start_of_no_interaction_",
}

# ---- PR CURVE (IoU threshold sweep) ----
PR_TMIN = 0.00
PR_TMAX = 1.00
PR_STEPS = 21  # 0.00, 0.05, ..., 1.00

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


# =============================================================================
# ============================ INTERVAL CONSTRUCTION ===========================
# =============================================================================


def collect_events(labels: Dict[str, Any]) -> List[Tuple[int, Dict[str, Any]]]:
    """
    Collect raw frame payloads as (frame_idx, payload) sorted by frame index.
    """
    events = []
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
        events.append((fidx, payload))
    events.sort(key=lambda x: x[0])
    return events


def intervals_from_labels(
    labels: Dict[str, Any],
    start_names: set,
    stop_names: set,
    kind: str,
    use_pred_boxes: bool,
) -> List[Dict[str, Any]]:
    """
    Build inclusive intervals from boundary labels.
      kind='gt'   → (Start of TTI) .. (End of TTI_)
      kind='pred' → (Start of TTI) .. (Start of No Interaction)
    Each interval dict has: {'start','end','start_box','end_box'}
      • For GT   we read boxes from 'boundingBox'
      • For Pred we read boxes from 'tti_bounding_box'
    Notes:
      - If an end/stop is missing a box, end_box=None (we will carry start_box)
      - If multiple starts occur before a stop, we close/reopen in order (sweep)
      - Intervals are INCLUSIVE on both ends as requested
    """
    evs = collect_events(labels)

    open_start = None
    start_box = None
    out: List[Dict[str, Any]] = []

    for fidx, payload in evs:
        objects = payload.get("objects", []) if isinstance(payload, dict) else []
        for obj in objects:
            name = _norm_text(obj.get("name", ""))
            # choose field for the box
            if use_pred_boxes:
                box = obj.get("tti_bounding_box") or obj.get("tti_boundingBox")
            else:
                box = obj.get("boundingBox") or obj.get("bounding_box")

            # START
            if name in start_names:
                open_start = fidx
                start_box = (
                    box
                    if (
                        isinstance(box, dict)
                        and all(k in box for k in ("x", "y", "w", "h"))
                    )
                    else None
                )

            # END/STOP
            elif name in stop_names and open_start is not None:
                end_frame = fidx  # inclusive
                end_box = (
                    box
                    if (
                        isinstance(box, dict)
                        and all(k in box for k in ("x", "y", "w", "h"))
                    )
                    else None
                )
                out.append(
                    {
                        "start": open_start,
                        "end": end_frame,
                        "start_box": start_box,
                        "end_box": end_box,
                    }
                )
                open_start, start_box = None, None

    # If sequence ended while active, close at the last seen frame in labels (inclusive)
    if open_start is not None and evs:
        last_frame = evs[-1][0]
        out.append(
            {
                "start": open_start,
                "end": max(open_start, last_frame),
                "start_box": start_box,
                "end_box": None,
            }
        )

    return out


# =============================================================================
# =========================== PER-FRAME BOX & MASKS ===========================
# =============================================================================


def lerp(a: float, b: float, t: float) -> float:
    return a + (b - a) * t


def interpolate_box(
    sb: Optional[Dict[str, float]],
    eb: Optional[Dict[str, float]],
    s: int,
    e: int,
    f: int,
) -> Optional[Dict[str, float]]:
    """
    Linear interpolation between start/end boxes over frames [s..e] inclusive.
    If only one box exists, carry it.
    """
    if s == e:
        return sb or eb
    if sb and eb:
        t = (f - s) / float(e - s)
        return {
            "x": lerp(float(sb["x"]), float(eb["x"]), t),
            "y": lerp(float(sb["y"]), float(eb["y"]), t),
            "w": lerp(float(sb["w"]), float(eb["w"]), t),
            "h": lerp(float(sb["h"]), float(eb["h"]), t),
        }
    return sb or eb


def rasterize_boxes_norm(boxes: List[Dict[str, float]], W: int, H: int) -> np.ndarray:
    """
    Rasterize normalized xywh boxes into a binary mask (H,W).
    """
    m = np.zeros((H, W), dtype=np.uint8)
    for b in boxes:
        if not all(k in b for k in ("x", "y", "w", "h")):
            continue
        x1 = int(round(float(b["x"]) * W))
        y1 = int(round(float(b["y"]) * H))
        x2 = int(round((float(b["x"]) + float(b["w"])) * W))
        y2 = int(round((float(b["y"]) + float(b["h"])) * H))
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(W, x2), min(H, y2)
        if x2 > x1 and y2 > y1:
            m[y1:y2, x1:x2] = 1
    return m


def build_masks_from_intervals(
    intervals: List[Dict[str, Any]], frames: List[int], W: int, H: int
) -> Dict[int, np.ndarray]:
    """
    For each requested frame, union all interval boxes active at that frame (with interpolation).
    Returns: {frame_idx: mask(H,W)}
    """
    out: Dict[int, np.ndarray] = {}
    for f in frames:
        boxes_f: List[Dict[str, float]] = []
        for iv in intervals:
            if iv["start"] <= f <= iv["end"]:
                b = interpolate_box(
                    iv.get("start_box"), iv.get("end_box"), iv["start"], iv["end"], f
                )
                if b:
                    boxes_f.append(b)
        out[f] = (
            rasterize_boxes_norm(boxes_f, W, H)
            if boxes_f
            else np.zeros((H, W), np.uint8)
        )
    return out


# =============================================================================
# =============================== METRICS & PLOTS =============================
# =============================================================================


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


def build_pr_curve(rows: List[Dict[str, Any]], out_png: str):
    """
    Frame-level PR using IoU as a quality score.
      gt_pos := gt_area > 0
      pred_pos := pred_area > 0
      For τ in [0..1]:
        TP := gt_pos & pred_pos & (IoU >= τ)
        FP := pred_pos & (IoU < τ)
        FN := gt_pos & (~pred_pos)
    """
    if not rows:
        return
    gt_pos = np.array([r["gt_area"] > 0 for r in rows], dtype=bool)
    pred_pos = np.array([r["pred_area"] > 0 for r in rows], dtype=bool)
    ious = np.array([r["iou"] for r in rows], dtype=float)

    thresholds = np.linspace(PR_TMIN, PR_TMAX, PR_STEPS)
    P, R, F1 = [], [], []
    for t in thresholds:
        tp = int(np.sum(gt_pos & pred_pos & (ious >= t)))
        fp = int(np.sum(pred_pos & (ious < t)))
        fn = int(np.sum(gt_pos & (~pred_pos)))
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (2 * prec * rec / (prec + rec)) if (prec + rec) > 0 else 0.0
        P.append(prec)
        R.append(rec)
        F1.append(f1)

    plt.figure(figsize=(7, 5))
    plt.plot(thresholds, P, label="Precision")
    plt.plot(thresholds, R, label="Recall")
    plt.plot(thresholds, F1, label="F1")
    plt.xlabel("IoU threshold (τ)")
    plt.ylabel("Score")
    plt.title("PR / F1 vs IoU Threshold (Frame-level)")
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_png, dpi=220)
    plt.close()


def save_sample_confusion_table(rows: List[Dict[str, Any]], out_png: str):
    if not rows:
        return
    informative = [r for r in rows if (r["gt_area"] + r["pred_area"]) > 0]
    sample = informative or rows
    sample.sort(key=lambda r: r["iou"])
    r = sample[len(sample) // 2]

    fig, ax = plt.subplots(figsize=(4, 3))
    ax.axis("off")
    table_data = [
        ["", "Pred TTI", "Pred No-TTI"],
        ["GT TTI", f"TP = {r['tp']}", f"FN = {r['fn']}"],
        ["GT No-TTI", f"FP = {r['fp']}", f"TN = {r['tn']}"],
    ]
    tbl = ax.table(cellText=table_data, loc="center", cellLoc="center")
    tbl.scale(1.2, 1.6)
    ax.set_title(f"Sample Confusion — {r['video']} f{r['frame']} (IoU={r['iou']:.2f})")
    plt.tight_layout()
    plt.savefig(out_png, dpi=160)
    plt.close()


# =============================================================================
# ================================ MAIN PIPELINE ==============================
# =============================================================================


def main():
    ap = argparse.ArgumentParser("Range-based Pixel Evaluation (GT vs Pred)")
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

    print("[INIT] Scanning directories...")
    print(f"  GT dir:   {args.gt_dir}")
    print(f"  Pred dir: {args.pred_dir}")
    print(f"  Out dir:  {args.out_dir}")

    gt_files = list_jsons(args.gt_dir)
    pred_files = list_jsons(args.pred_dir, prefix="pred_")

    if not gt_files:
        print(f"[ERROR] No GT JSONs found in {args.gt_dir}")
        return
    if not pred_files:
        print(f"[ERROR] No Pred JSONs (pred_*.json) found in {args.pred_dir}")
        return

    print(f"[INIT] Found {len(gt_files)} GT files and {len(pred_files)} Pred files")

    # key by normalized stem (strip 'pred_' and extension; lower/collapse spaces)
    def norm_key(p: str) -> str:
        base = os.path.basename(p)
        stem, _ = os.path.splitext(base)
        if stem.startswith("pred_"):
            stem = stem[5:]
        return _norm_text(stem)

    gt_map = {norm_key(p): p for p in gt_files}
    pred_map = {norm_key(p): p for p in pred_files}
    common = sorted(set(gt_map.keys()) & set(pred_map.keys()))
    if not common:
        print("[ERROR] No matching GT / Pred JSON pairs after normalization.")
        return

    print(f"[INIT] Matched {len(common)} video pairs")
    print(f"[INIT] Canvas size: {args.canvas_w}x{args.canvas_h}")
    print(f"\n{'=' * 60}")
    print("STARTING PROCESSING")
    print(f"{'=' * 60}")

    all_rows: List[Dict[str, Any]] = []
    per_video: List[Dict[str, Any]] = []

    for idx, key in enumerate(common, 1):
        gt_path = gt_map[key]
        pred_path = pred_map[key]
        video_name = os.path.splitext(os.path.basename(gt_path))[0]
        print(f"\n[PAIR {idx}/{len(common)}] {video_name}")

        print("  → Loading JSONs...")
        gt_root = load_json_any(gt_path)
        pred_root = load_json_any(pred_path)
        if not gt_root or not pred_root:
            print("  [skip] invalid JSON format")
            continue

        print("  → Extracting labels...")
        gt_labels = get_labels_map(gt_root)
        pred_labels = get_labels_map(pred_root)
        if not gt_labels or not pred_labels:
            print("  [skip] labels missing")
            continue

        # Build inclusive intervals
        print("  → Building intervals...")
        gt_intervals = intervals_from_labels(
            gt_labels,
            start_names=GT_START_LABELS,
            stop_names=GT_END_LABELS,
            kind="gt",
            use_pred_boxes=False,
        )
        pred_intervals = intervals_from_labels(
            pred_labels,
            start_names=PRED_START_LABELS,
            stop_names=PRED_STOP_LABELS,
            kind="pred",
            use_pred_boxes=True,
        )

        print(
            f"     GT intervals: {len(gt_intervals)}, Pred intervals: {len(pred_intervals)}"
        )

        if not gt_intervals and not pred_intervals:
            print("  [skip] no intervals in GT or Pred")
            continue

        # Union of frames across all intervals (inclusive)
        print("  → Computing frame union...")
        frame_set: set = set()
        for iv in gt_intervals:
            frame_set.update(range(iv["start"], iv["end"] + 1))
        for iv in pred_intervals:
            frame_set.update(range(iv["start"], iv["end"] + 1))
        frames = sorted(frame_set)
        if not frames:
            print("  [skip] empty frame set")
            continue

        print(
            f"     Total frames to process: {len(frames)} (range: {min(frames)}-{max(frames)})"
        )

        # Build per-frame masks on a fixed canvas
        print(f"  → Building GT masks ({len(frames)} frames)...")
        gt_masks = build_masks_from_intervals(
            gt_intervals, frames, args.canvas_w, args.canvas_h
        )
        print(f"  → Building Pred masks ({len(frames)} frames)...")
        pred_masks = build_masks_from_intervals(
            pred_intervals, frames, args.canvas_w, args.canvas_h
        )

        # Per-frame metrics
        print("  → Computing per-frame metrics...")
        rows_this_video = []
        for frame_idx, f in enumerate(frames):
            if frame_idx % 100 == 0 and frame_idx > 0:
                print(f"     ... processed {frame_idx}/{len(frames)} frames")
            tp, fp, fn, tn, iou, dice = pixel_confusion_and_scores(
                gt_masks[f], pred_masks[f]
            )
            rows_this_video.append(
                {
                    "video": video_name,
                    "frame": f,
                    "tp": tp,
                    "fp": fp,
                    "fn": fn,
                    "tn": tn,
                    "iou": iou,
                    "dice": dice,
                    "gt_area": int(gt_masks[f].sum()),
                    "pred_area": int(pred_masks[f].sum()),
                }
            )

        # Aggregate per video (exclude frames with no positive pixels in both)
        print("  → Aggregating video metrics...")
        dfv = pd.DataFrame(rows_this_video)
        informative = (dfv["tp"] + dfv["fp"] + dfv["fn"]) > 0
        if informative.any():
            iou_mean = float(dfv.loc[informative, "iou"].mean())
            iou_std = float(dfv.loc[informative, "iou"].std(ddof=0))
            dice_mean = float(dfv.loc[informative, "dice"].mean())
            dice_std = float(dfv.loc[informative, "dice"].std(ddof=0))
            used = int(informative.sum())
        else:
            iou_mean = iou_std = dice_mean = dice_std = 0.0
            used = 0

        per_video.append(
            {
                "video": video_name,
                "frames_used": used,
                "iou_mean": iou_mean,
                "iou_std": iou_std,
                "dice_mean": dice_mean,
                "dice_std": dice_std,
            }
        )

        all_rows.extend(rows_this_video)
        print(f"  ✓ Completed {video_name} (IoU: {iou_mean:.3f}, {used} frames)")

    print(f"\n{'=' * 60}")
    print("SAVING RESULTS")
    print(f"{'=' * 60}")

    # Save per-frame metrics
    print(f"[OUTPUT] Creating per-frame DataFrame ({len(all_rows)} rows)...")
    df_all = pd.DataFrame(all_rows).sort_values(["video", "frame"])
    per_frame_csv = os.path.join(args.out_dir, "pixel_metrics_per_frame.csv")
    print(f"[OUTPUT] Writing {per_frame_csv}...")
    df_all.to_csv(per_frame_csv, index=False)
    print(f"[OK] {per_frame_csv}")

    # Save per-video metrics
    print(f"[OUTPUT] Creating per-video DataFrame ({len(per_video)} rows)...")
    df_vid = pd.DataFrame(per_video).sort_values(["video"])
    per_video_csv = os.path.join(args.out_dir, "pixel_metrics_per_video.csv")
    print(f"[OUTPUT] Writing {per_video_csv}...")
    df_vid.to_csv(per_video_csv, index=False)
    print(f"[OK] {per_video_csv}")

    # PR curve
    pr_png = os.path.join(args.out_dir, "pr_curve.png")
    print("[OUTPUT] Generating PR curve...")
    build_pr_curve(df_all.to_dict("records"), pr_png)
    print(f"[OK] {pr_png}")

    # Sample confusion image
    conf_png = os.path.join(args.out_dir, "sample_confusion_table.png")
    print("[OUTPUT] Generating confusion table...")
    save_sample_confusion_table(df_all.to_dict("records"), conf_png)
    print(f"[OK] {conf_png}")

    print(f"\n{'=' * 60}")
    print("✅ COMPLETE")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
