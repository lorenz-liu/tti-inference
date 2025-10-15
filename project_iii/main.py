#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Project III — Interaction-Level Detection Analysis
Author: Lorenz

What this script does
---------------------
1) Reads ground-truth JSONs with temporal labels:
     - "Start of TTI", "End of TTI", "Start of No Interaction", "End of No Interaction"
2) Reads model predictions from inferences:
     - Prefer JSON if available (pred_*.json), otherwise fall back to MP4 (pred_*.mp4)
3) Builds GT interaction intervals per video (contiguous frames where TTI is "on")
4) Builds predicted-per-frame TTI booleans and predicted intervals
5) For each GT interval, computes the detected-frame ratio:
       ratio = (# predicted TTI frames within the GT interval) / (interval length)
6) Sweeps thresholds (default 0.05..1.0) to compute Precision/Recall/F1:
     - TP = GT intervals with ratio >= threshold
     - FP = predicted intervals that do NOT overlap any GT interval
     - FN = GT intervals with ratio < threshold
7) Outputs:
     - CSV of per-interval ratios
     - CSV of PR/F1 across thresholds
     - Histogram of ratios, PR curve, F1 vs threshold plot

Usage
-----
python3 this.py \
  --gt_dir /cluster/projects/madanigroup/lorenz/tti/ground_truths \
  --inf_dir /cluster/projects/madanigroup/lorenz/tti/inferences \
  --out_dir /cluster/projects/madanigroup/lorenz/tti/project_iii_out \
  --fps 30

Notes
-----
- If FPS varies by video and is needed, you can omit --fps; the script will try to read FPS from the MP4 when falling back.
- Ground truth format is expected to be similar to the example provided (Labelbox-like).
"""

import argparse
import glob
import json
import os
import re
from collections import defaultdict

import cv2
import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd


# -------------------
# Helpers: paths/names
# -------------------
def norm_key(name: str) -> str:
    """
    Normalize a video name for matching across gt and inference:
    - strip extension
    - lower, normalize spaces
    - remove 'pred_' prefix if present
    """
    base = os.path.basename(name)
    stem, _ = os.path.splitext(base)
    if stem.startswith("pred_"):
        stem = stem[5:]
    return re.sub(r"\s+", " ", stem.strip().lower())


def list_json(folder):
    return sorted(glob.glob(os.path.join(folder, "*.json")))


def list_inference_assets(inf_dir):
    inf_jsons = sorted(
        [
            p
            for p in glob.glob(os.path.join(inf_dir, "*.json"))
            if os.path.basename(p).startswith("pred_")
        ]
    )
    inf_videos = sorted(
        [
            p
            for p in glob.glob(os.path.join(inf_dir, "*.*"))
            if os.path.splitext(p)[1].lower() in [".mp4", ".mov", ".avi"]
            and os.path.basename(p).startswith("pred_")
        ]
    )
    return inf_jsons, inf_videos


# -------------------------
# Ground-truth JSON parsing
# -------------------------
GT_START_TAGS = {"start of tti", "start_of_tti"}
GT_END_TAGS = {"end of tti", "end_of_tti", "end_of_tti_"}

# Optional tags (not strictly required)
NO_INT_START = {
    "start of no interaction",
    "start_of_no_interaction",
    "start_of_no_interaction_",
}
NO_INT_END = {"end of no interaction", "end_of_no_interaction"}


def extract_gt_events(gt_json_path):
    """
    Parse Labelbox-like ground-truth JSON.
    Returns a sorted list of (frame_idx, event_type) where event_type in {'start','end'} for TTI.
    """
    with open(gt_json_path, "r") as f:
        data = json.load(f)

    # data["..."]["labels"] is a dict keyed by frame-number strings
    # Each has "objects" which may include "name" and "frame"
    events = []
    # find first (and only) data_units key
    du = data.get("data_units", {})
    if not du:
        # Possibly a different flat format, try to harvest top-level frames
        # Fallback: look for generic "labels" mapping
        labels = data.get("labels", {})
        iterable = labels.items()
    else:
        # choose the only data_unit present
        du_key = next(iter(du.keys()))
        labels = du[du_key].get("labels", {})
        iterable = labels.items()

    for frame_key, frame_payload in iterable:
        try:
            frame_idx = int(frame_key)
        except Exception:
            # some exports put frame idx also inside each object; still try best-effort
            frame_idx = None

        objects = (
            frame_payload.get("objects", []) if isinstance(frame_payload, dict) else []
        )
        for obj in objects:
            name = obj.get("name", "").strip().lower().replace("  ", " ")
            fidx = obj.get("frame", frame_idx)
            if fidx is None:
                continue
            if name in GT_START_TAGS:
                events.append((int(fidx), "start"))
            elif name in GT_END_TAGS:
                events.append((int(fidx), "end"))
            # No-Interaction tags are not strictly needed for interval building

    events.sort(key=lambda x: x[0])
    return events


def build_gt_intervals(events):
    """
    Build contiguous GT TTI intervals from start/end events.
    Supports nested/overlapping via counter; any frames with counter>0 considered TTI-on.
    Returns list of (start_frame, end_frame) inclusive.
    """
    if not events:
        return []

    # Build a frame->delta map then sweep
    deltas = defaultdict(int)
    for f, ev in events:
        if ev == "start":
            deltas[f] += 1
        elif ev == "end":
            # common labeling assumes inclusive end at 'end' frame; we treat end at that frame
            deltas[f + 1] -= 1  # close AFTER end frame to keep end inclusive

    # Convert deltas to intervals by sweep
    active = 0
    intervals = []
    current_start = None

    for f in sorted(deltas.keys()):
        prev_active = active
        active += deltas[f]
        if prev_active <= 0 and active > 0:
            # turned on
            current_start = f
        elif prev_active > 0 and active <= 0:
            # turned off at f-1
            if current_start is not None:
                intervals.append((current_start, f - 1))
                current_start = None

    # In case still active until last annotated change (cannot know end), close conservatively
    if current_start is not None:
        # choose last known change frame as end (best-effort)
        last_change = max(deltas.keys())
        intervals.append((current_start, last_change))

    # Filter out invalid
    intervals = [(s, e) for s, e in intervals if e >= s]
    return intervals


# --------------------------------
# Prediction parsing (JSON or MP4)
# --------------------------------
def load_pred_bool_from_json(pred_json_path, total_frames_hint=None):
    """
    Try to parse a generic per-frame prediction JSON into a boolean array 'tti_pred[frame]=1/0'.
    Supports a few common shapes; falls back to None if unrecognized.
    """
    try:
        with open(pred_json_path, "r") as f:
            data = json.load(f)
    except Exception:
        return None

    # Heuristic 1: { "frames": { "0": {"tti": true}, ... } }
    if isinstance(data, dict):
        frames_map = None
        if "frames" in data and isinstance(data["frames"], dict):
            frames_map = data["frames"]
        elif "frame_predictions" in data and isinstance(
            data["frame_predictions"], dict
        ):
            frames_map = data["frame_predictions"]

        if frames_map is not None:
            max_idx = -1
            pred = {}
            for k, v in frames_map.items():
                try:
                    idx = int(k)
                except Exception:
                    continue
                # Common boolean fields
                val = 0
                if isinstance(v, dict):
                    # any field indicating contact?
                    for key in [
                        "tti",
                        "has_contact",
                        "contact",
                        "pred",
                        "mask_present",
                    ]:
                        if key in v:
                            val = 1 if (v[key] in (1, True, "1", "true", "True")) else 0
                            break
                    # Or non-empty detections list
                    if val == 0 and any(
                        isinstance(v.get(k2), list) and len(v.get(k2)) > 0
                        for k2 in ["detections", "bboxes", "instances", "masks"]
                    ):
                        val = 1
                elif isinstance(v, (int, bool)):
                    val = 1 if int(v) != 0 else 0
                pred[idx] = val
                max_idx = max(max_idx, idx)
            if max_idx >= 0:
                length = total_frames_hint or (max_idx + 1)
                arr = np.zeros(length, dtype=np.uint8)
                for i, v in pred.items():
                    if 0 <= i < length:
                        arr[i] = 1 if v else 0
                return arr

        # Heuristic 2: flat list of frame indices predicted as TTI
        if "tti_frames" in data and isinstance(data["tti_frames"], list):
            idxs = [
                int(x)
                for x in data["tti_frames"]
                if isinstance(x, (int, str)) and str(x).isdigit()
            ]
            length = total_frames_hint or (max(idxs) + 1 if idxs else 0)
            arr = np.zeros(length, dtype=np.uint8)
            for i in idxs:
                if 0 <= i < length:
                    arr[i] = 1
            return arr

    return None


def load_pred_bool_from_video(pred_video_path, min_nonzero_ratio=1e-4):
    """
    Fallback: derive TTI presence per frame from a prediction video.
    Assumes non-black pixels indicate TTI. Returns (bool_array, fps).
    """
    cap = cv2.VideoCapture(pred_video_path)
    if not cap.isOpened():
        return None, None
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    arr = np.zeros(n, dtype=np.uint8)

    for i in range(n):
        ok, frame = cap.read()
        if not ok:
            break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        nz = int(np.count_nonzero(gray))
        ratio = nz / gray.size
        arr[i] = 1 if ratio >= min_nonzero_ratio else 0

    cap.release()
    return arr, fps


def extract_pred_intervals(pred_bool):
    """
    From per-frame boolean array, return list of contiguous predicted intervals (start,end) inclusive.
    """
    intervals = []
    if pred_bool is None or len(pred_bool) == 0:
        return intervals
    on = False
    start = 0
    for i, v in enumerate(pred_bool):
        if not on and v == 1:
            on = True
            start = i
        elif on and v == 0:
            intervals.append((start, i - 1))
            on = False
    if on:
        intervals.append((start, len(pred_bool) - 1))
    return intervals


# -----------------------
# Core metrics computation
# -----------------------
def ratio_for_interval(pred_bool, interval):
    s, e = interval
    if e < s or s < 0:
        return 0.0
    e = min(e, len(pred_bool) - 1)
    if e < s:
        return 0.0
    seg = pred_bool[s : e + 1]
    if seg.size == 0:
        return 0.0
    return float(np.sum(seg)) / float(seg.size)


def count_fp(pred_intervals, gt_intervals):
    """
    FP = predicted intervals that do NOT overlap any GT interval (temporal overlap).
    """

    def overlaps(a, b):
        return not (a[1] < b[0] or b[1] < a[0])

    fp = 0
    for p in pred_intervals:
        if not any(overlaps(p, g) for g in gt_intervals):
            fp += 1
    return fp


def sweep_thresholds(ratios, pred_intervals, gt_intervals, thresholds):
    """
    Compute Precision/Recall/F1 across thresholds.
    TP = #GT intervals whose ratio >= t
    FN = #GT intervals whose ratio <  t
    FP = predicted intervals with no overlap to any GT interval (constant across t)  [conservative baseline]
    """
    fp_const = count_fp(pred_intervals, gt_intervals)
    out = []
    for t in thresholds:
        tp = int(np.sum(np.array(ratios) >= t))
        fn = len(ratios) - tp
        fp = fp_const
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (2 * prec * rec / (prec + rec)) if (prec + rec) > 0 else 0.0
        out.append((t, prec, rec, f1, tp, fp, fn))
    return out


# -------------
# Main pipeline
# -------------
def main(args):
    os.makedirs(args.out_dir, exist_ok=True)

    gt_jsons = list_json(args.gt_dir)
    inf_jsons, inf_videos = list_inference_assets(args.inf_dir)

    gt_map = {norm_key(p): p for p in gt_jsons}
    inf_json_map = {norm_key(p): p for p in inf_jsons}
    inf_vid_map = {norm_key(p): p for p in inf_videos}

    keys = sorted(
        set(gt_map.keys()) & (set(inf_json_map.keys()) | set(inf_vid_map.keys()))
    )
    if not keys:
        print(
            "No matching videos between ground_truths and inferences. Check filenames."
        )
        return

    per_interval_rows = []
    pr_rows = []

    for k in keys:
        gt_path = gt_map[k]
        pred_json = inf_json_map.get(k, None)
        pred_video = inf_vid_map.get(k, None)

        video_name = os.path.splitext(os.path.basename(gt_path))[0]
        print(f"[Video] {video_name}")

        # 1) Build GT intervals
        events = extract_gt_events(gt_path)
        gt_intervals = build_gt_intervals(events)
        if not gt_intervals:
            print("  - No GT intervals found, skipping.")
            continue

        # 2) Build prediction bool array
        pred_bool = None
        total_frames_hint = None

        if pred_video:
            cap = cv2.VideoCapture(pred_video)
            total_frames_hint = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
            fps_from_vid = cap.get(cv2.CAP_PROP_FPS) or args.fps
            cap.release()
        else:
            fps_from_vid = args.fps

        if pred_json:
            pred_bool = load_pred_bool_from_json(pred_json, total_frames_hint)

        if pred_bool is None:
            if pred_video:
                pred_bool, _fps = load_pred_bool_from_video(pred_video)
            else:
                print("  - No usable prediction source (JSON/MP4), skipping.")
                continue

        # 3) Ratios for each GT interval
        ratios = []
        for s, e in gt_intervals:
            r = ratio_for_interval(pred_bool, (s, e))
            ratios.append(r)
            per_interval_rows.append(
                {
                    "video": video_name,
                    "gt_start": s,
                    "gt_end": e,
                    "length_frames": (e - s + 1),
                    "detected_frames": int(round(r * (e - s + 1))),
                    "ratio": r,
                }
            )

        # 4) Predicted intervals for FP baseline
        pred_intervals = extract_pred_intervals(pred_bool)

        # 5) Threshold sweep
        thresholds = np.round(np.linspace(args.t_min, args.t_max, args.t_steps), 3)
        stats = sweep_thresholds(ratios, pred_intervals, gt_intervals, thresholds)

        for t, p, r, f1, tp, fp, fn in stats:
            pr_rows.append(
                {
                    "video": video_name,
                    "threshold": t,
                    "precision": p,
                    "recall": r,
                    "f1": f1,
                    "tp": tp,
                    "fp": fp,
                    "fn": fn,
                    "gt_interactions": len(gt_intervals),
                    "pred_interactions": len(pred_intervals),
                }
            )

        # 6) Plots (per-video)
        # Histogram of ratios
        plt.figure(figsize=(6, 4))
        plt.hist(ratios, bins=20, edgecolor="black")
        plt.xlabel("Detected-frame ratio within GT interaction")
        plt.ylabel("Count")
        plt.title(f"Ratio Histogram — {video_name}")
        plt.tight_layout()
        plt.savefig(os.path.join(args.out_dir, f"{video_name}_ratio_hist.png"))
        plt.close()

        # PR & F1 curves
        th = [x[0] for x in stats]
        prec = [x[1] for x in stats]
        rec = [x[2] for x in stats]
        f1s = [x[3] for x in stats]

        plt.figure(figsize=(6, 4))
        plt.plot(th, prec, label="Precision")
        plt.plot(th, rec, label="Recall")
        plt.plot(th, f1s, label="F1")
        plt.xlabel("Threshold (min ratio to count GT interaction as detected)")
        plt.ylabel("Score")
        plt.title(f"PR/F1 vs Threshold — {video_name}")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(args.out_dir, f"{video_name}_pr_f1.png"))
        plt.close()

    # 7) Save CSVs (aggregated)
    df_intervals = pd.DataFrame(per_interval_rows)
    df_pr = pd.DataFrame(pr_rows)

    intervals_csv = os.path.join(args.out_dir, "interaction_ratios.csv")
    pr_csv = os.path.join(args.out_dir, "pr_sweep.csv")
    df_intervals.to_csv(intervals_csv, index=False)
    df_pr.to_csv(pr_csv, index=False)

    # 8) Global plots (across videos)
    if not df_intervals.empty:
        plt.figure(figsize=(6, 4))
        plt.hist(df_intervals["ratio"].values, bins=30, edgecolor="black")
        plt.xlabel("Detected-frame ratio within GT interaction")
        plt.ylabel("Count")
        plt.title("Global Ratio Histogram (All Videos)")
        plt.tight_layout()
        plt.savefig(os.path.join(args.out_dir, "global_ratio_hist.png"))
        plt.close()

    if not df_pr.empty:
        # Aggregate by threshold: mean precision/recall/f1 across videos
        agg = (
            df_pr.groupby("threshold")[["precision", "recall", "f1"]]
            .mean()
            .reset_index()
        )
        plt.figure(figsize=(6, 4))
        plt.plot(agg["threshold"], agg["precision"], label="Precision (mean)")
        plt.plot(agg["threshold"], agg["recall"], label="Recall (mean)")
        plt.plot(agg["threshold"], agg["f1"], label="F1 (mean)")
        plt.xlabel("Threshold")
        plt.ylabel("Score")
        plt.title("PR/F1 vs Threshold (Mean Across Videos)")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(args.out_dir, "global_pr_f1.png"))
        plt.close()

    print(f"✅ Saved:\n  - {intervals_csv}\n  - {pr_csv}\n  - plots in {args.out_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Project III — Interaction-Level Detection Analysis"
    )
    parser.add_argument(
        "--gt_dir", required=True, help="Directory with ground-truth JSONs"
    )
    parser.add_argument(
        "--inf_dir",
        required=True,
        help="Directory with inference outputs (pred_*.json / pred_*.mp4)",
    )
    parser.add_argument(
        "--out_dir", required=True, help="Directory to save CSVs and plots"
    )
    parser.add_argument(
        "--fps",
        type=float,
        default=30.0,
        help="Fallback FPS if needed (for reference only)",
    )
    parser.add_argument(
        "--t_min", type=float, default=0.05, help="Min threshold for ratio sweep"
    )
    parser.add_argument(
        "--t_max", type=float, default=1.0, help="Max threshold for ratio sweep"
    )
    parser.add_argument(
        "--t_steps",
        type=int,
        default=20,
        help="Number of thresholds between t_min and t_max",
    )
    args = parser.parse_args()
    main(args)
