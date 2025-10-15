#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Project III — Interaction-Level Detection Analysis
Author: Lorenz  |  October 2025

Purpose
-------
Transition from frame-level to interaction-level evaluation.
Determine what proportion of frames within a ground-truth interaction
must be detected by the AI for that interaction to count as detected.

Outputs
-------
  /
    ├── interaction_ratios.csv     # Per-interaction ratios
    ├── pr_sweep.csv               # Precision/Recall/F1 per threshold
    ├── *_ratio_hist.png           # Histograms
    ├── *_pr_f1.png                # PR/F1 per video
    ├── global_ratio_hist.png
    └── global_pr_f1.png

Usage
-----
python3 main.py
"""

# ==========================================================
# CONFIGURATION (EDIT THESE DEFAULTS AS NEEDED)
# ==========================================================

GT_DIR = "/cluster/projects/madanigroup/lorenz/tti/ground_truths"
INF_DIR = "/cluster/projects/madanigroup/lorenz/tti/inferences"
OUT_DIR = "./"

# Default analysis parameters
DEFAULT_FPS = 30.0
THRESH_MIN = 0.05
THRESH_MAX = 1.0
THRESH_STEPS = 20
MIN_NONZERO_TTI = 1e-4  # ratio threshold for “non-black” detection from videos

# ==========================================================
# CORE SCRIPT STARTS HERE
# ==========================================================

import glob
import json
import os
import re
from collections import defaultdict

import cv2
import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt


# -------------------
# Helpers: paths/names
# -------------------
def norm_key(name: str) -> str:
    """Normalize filename (lowercase, remove 'pred_', ignore extension)."""
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
# Ground-truth parsing
# -------------------------
def _norm_event_name(s: str) -> str:
    return re.sub(r"\s+", " ", (s or "").strip().lower())


# Accept common variants (trailing underscores/spaces handled by _norm_event_name)
GT_START_TAGS = {"start of tti", "start_of_tti"}
GT_END_TAGS = {"end of tti", "end_of_tti", "end_of_tti_"}


def extract_gt_events(gt_json_path):
    """
    Parse Labelbox-like ground-truth JSON.
    Robust to:
      - top-level list (common export) or dict
      - 'labels' under data_units[...] or at top-level
      - trailing spaces / underscores in names (e.g., 'End of TTI ')
    Returns: sorted list[(frame_idx:int, 'start'|'end')]
    """
    import json

    with open(gt_json_path, "r") as f:
        data = json.load(f)

    # Top-level can be a list (pick first) or dict
    root = data[0] if isinstance(data, list) and data else data

    # Prefer labels nested under data_units[...] if present
    labels = None
    if isinstance(root, dict):
        du = root.get("data_units", {})
        if isinstance(du, dict) and du:
            du_key = next(iter(du.keys()))
            labels = du.get(du_key, {}).get("labels", None)

        # Fallback to top-level 'labels'
        if labels is None:
            labels = root.get("labels", None)

    if not isinstance(labels, dict):
        # Nothing we can parse
        return []

    events = []
    for frame_key, frame_payload in labels.items():
        # Some exports keep frame as the object.frame; still try key first
        try:
            frame_idx_default = int(frame_key)
        except Exception:
            frame_idx_default = None

        objects = (
            frame_payload.get("objects", []) if isinstance(frame_payload, dict) else []
        )
        for obj in objects:
            name = _norm_event_name(obj.get("name", ""))
            # prefer explicit 'frame' in object, fallback to key-derived index
            fidx = obj.get("frame", frame_idx_default)
            if fidx is None:
                continue
            if name in GT_START_TAGS:
                events.append((int(fidx), "start"))
            elif name in GT_END_TAGS:
                events.append((int(fidx), "end"))

    events.sort(key=lambda x: x[0])
    return events


def build_gt_intervals(events):
    """
    Convert start/end events to inclusive intervals (start,end).
    Supports multiple overlapping starts/ends via a sweep counter.
    'End' is treated as inclusive.
    """
    if not events:
        return []

    deltas = defaultdict(int)
    for f, ev in events:
        if ev == "start":
            deltas[int(f)] += 1
        elif ev == "end":
            # close after end frame to keep it inclusive
            deltas[int(f) + 1] -= 1

    active = 0
    current_start = None
    intervals = []

    for f in sorted(deltas.keys()):
        prev = active
        active += deltas[f]
        if prev <= 0 < active:
            current_start = f
        elif prev > 0 >= active and current_start is not None:
            intervals.append((current_start, f - 1))
            current_start = None

    if current_start is not None:
        # If labeling ended while active, close at last change frame - 1 (best effort)
        last_change = max(deltas.keys())
        intervals.append((current_start, last_change - 1))

    # sanitize
    intervals = [(int(s), int(e)) for s, e in intervals if e >= s]
    return intervals


# -------------------------
# Prediction loading
# -------------------------
def _norm(s: str) -> str:
    return re.sub(r"\s+", " ", (s or "").strip().lower())


def load_pred_bool_from_json(pred_json_path, total_frames_hint=None):
    """
    Labelbox-style prediction JSON (list or dict root).
    Marks a frame as TTI=1 iff any object in that frame has name/value == 'start_of_tti'.
    Ignores 'start_of_no_interaction'.
    """
    with open(pred_json_path, "r") as f:
        data = json.load(f)

    root = data[0] if isinstance(data, list) and data else data
    if not isinstance(root, dict):
        return None

    labels = None
    du = root.get("data_units", {})
    if isinstance(du, dict) and du:
        du_key = next(iter(du.keys()))
        labels = du.get(du_key, {}).get("labels", None)
    if labels is None:
        labels = root.get("labels", None)
    if not isinstance(labels, dict):
        return None

    pred_map = {}
    max_idx = -1

    for frame_key, payload in labels.items():
        # prefer explicit frame in payload, else key
        try:
            frame_idx_default = (
                int(payload.get("frame", int(frame_key)))
                if isinstance(payload, dict)
                else int(frame_key)
            )
        except Exception:
            frame_idx_default = None

        objs = payload.get("objects", []) if isinstance(payload, dict) else []
        frame_has_tti = 0

        for obj in objs:
            name = _norm(obj.get("name", ""))
            value = _norm(obj.get("value", ""))
            if (
                name == "start of tti"
                or name == "start_of_tti"
                or value == "start_of_tti"
            ):
                frame_has_tti = 1
                break  # one positive object is enough

        fidx = frame_idx_default
        if fidx is None and objs:
            try:
                fidx = int(objs[0].get("frame"))
            except Exception:
                fidx = None
        if fidx is None:
            continue

        pred_map[int(fidx)] = 1 if frame_has_tti else pred_map.get(int(fidx), 0)
        max_idx = max(max_idx, int(fidx))

    if max_idx < 0 and total_frames_hint is None:
        return None

    length = total_frames_hint if total_frames_hint is not None else (max_idx + 1)
    arr = np.zeros(length, dtype=np.uint8)
    for i, v in pred_map.items():
        if 0 <= i < length:
            arr[i] = 1 if v else 0
    return arr


def load_pred_bool_from_video(path):
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        return None, None
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS) or DEFAULT_FPS
    arr = np.zeros(n, np.uint8)
    for i in range(n):
        ok, f = cap.read()
        if not ok:
            break
        gray = cv2.cvtColor(f, cv2.COLOR_BGR2GRAY)
        arr[i] = 1 if np.count_nonzero(gray) / gray.size >= MIN_NONZERO_TTI else 0
    cap.release()
    return arr, fps


def extract_pred_intervals(pred_bool):
    intervals = []
    if pred_bool is None or len(pred_bool) == 0:
        return intervals
    on, start = False, 0
    for i, v in enumerate(pred_bool):
        if not on and v:
            on, start = True, i
        elif on and not v:
            intervals.append((start, i - 1))
            on = False
    if on:
        intervals.append((start, len(pred_bool) - 1))
    return intervals


# -------------------------
# Metrics
# -------------------------
def ratio_for_interval(pred_bool, interval):
    s, e = interval
    e = min(e, len(pred_bool) - 1)
    if e < s:
        return 0.0
    seg = pred_bool[s : e + 1]
    return float(np.sum(seg)) / float(len(seg)) if len(seg) else 0.0


def count_fp(pred_intervals, gt_intervals):
    def overlaps(a, b):
        return not (a[1] < b[0] or b[1] < a[0])

    return sum(not any(overlaps(p, g) for g in gt_intervals) for p in pred_intervals)


def sweep_thresholds(ratios, pred_intervals, gt_intervals, thresholds):
    fp_const = count_fp(pred_intervals, gt_intervals)
    out = []
    for t in thresholds:
        tp = int(np.sum(np.array(ratios) >= t))
        fn = len(ratios) - tp
        fp = fp_const
        prec = tp / (tp + fp) if (tp + fp) else 0
        rec = tp / (tp + fn) if (tp + fn) else 0
        f1 = (2 * prec * rec) / (prec + rec) if (prec + rec) else 0
        out.append((t, prec, rec, f1, tp, fp, fn))
    return out


# ==========================================================
# MAIN
# ==========================================================
def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    gt_jsons = list_json(GT_DIR)
    inf_jsons, inf_videos = list_inference_assets(INF_DIR)

    gt_map = {norm_key(p): p for p in gt_jsons}
    inf_json_map = {norm_key(p): p for p in inf_jsons}
    inf_vid_map = {norm_key(p): p for p in inf_videos}
    keys = sorted(set(gt_map) & (set(inf_json_map) | set(inf_vid_map)))
    if not keys:
        print("No matching ground-truth and inference videos found.")
        return

    interval_rows, pr_rows = [], []

    for k in keys:
        gt_path = gt_map[k]
        video_name = os.path.splitext(os.path.basename(gt_path))[0]
        pred_json, pred_video = inf_json_map.get(k), inf_vid_map.get(k)
        print(f"[Processing] {video_name}")

        # Ground truth intervals
        gt_intervals = build_gt_intervals(extract_gt_events(gt_path))
        if not gt_intervals:
            continue

        # Predictions
        total_hint, pred_bool = None, None
        if pred_video:
            cap = cv2.VideoCapture(pred_video)
            total_hint = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            cap.release()
        if pred_json:
            pred_bool = load_pred_bool_from_json(pred_json, total_hint)
        if pred_bool is None and pred_video:
            pred_bool, _ = load_pred_bool_from_video(pred_video)
        if pred_bool is None:
            continue

        ratios = [ratio_for_interval(pred_bool, iv) for iv in gt_intervals]
        for (s, e), r in zip(gt_intervals, ratios):
            interval_rows.append(
                dict(
                    video=video_name,
                    start=s,
                    end=e,
                    frames=e - s + 1,
                    detected=int(round(r * (e - s + 1))),
                    ratio=r,
                )
            )
        pred_intervals = extract_pred_intervals(pred_bool)
        thresholds = np.round(np.linspace(THRESH_MIN, THRESH_MAX, THRESH_STEPS), 3)
        stats = sweep_thresholds(ratios, pred_intervals, gt_intervals, thresholds)
        for t, p, r, f1, tp, fp, fn in stats:
            pr_rows.append(
                dict(
                    video=video_name,
                    threshold=t,
                    precision=p,
                    recall=r,
                    f1=f1,
                    tp=tp,
                    fp=fp,
                    fn=fn,
                )
            )

        # Per-video plots
        plt.figure(figsize=(6, 4))
        plt.hist(ratios, bins=20, edgecolor="black")
        plt.xlabel("Detected-frame ratio")
        plt.ylabel("Count")
        plt.title(f"Ratio Histogram — {video_name}")
        plt.tight_layout()
        plt.savefig(os.path.join(OUT_DIR, f"{video_name}_ratio_hist.png"))
        plt.close()

        tvals, pvals, rvals, fvals = zip(
            *[(t, p, r, f1) for t, p, r, f1, _, _, _ in stats]
        )
        plt.figure(figsize=(6, 4))
        plt.plot(tvals, pvals, label="Precision")
        plt.plot(tvals, rvals, label="Recall")
        plt.plot(tvals, fvals, label="F1")
        plt.xlabel("Threshold (ratio)")
        plt.ylabel("Score")
        plt.title(f"PR/F1 — {video_name}")
        plt.legend()
        plt.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(OUT_DIR, f"{video_name}_pr_f1.png"))
        plt.close()

    # Save CSVs
    df_int, df_pr = pd.DataFrame(interval_rows), pd.DataFrame(pr_rows)
    df_int.to_csv(os.path.join(OUT_DIR, "interaction_ratios.csv"), index=False)
    df_pr.to_csv(os.path.join(OUT_DIR, "pr_sweep.csv"), index=False)

    # Global plots
    if not df_int.empty:
        plt.figure(figsize=(6, 4))
        plt.hist(df_int["ratio"], bins=30, edgecolor="black")
        plt.xlabel("Detected-frame ratio")
        plt.ylabel("Count")
        plt.title("Global Ratio Histogram")
        plt.tight_layout()
        plt.savefig(os.path.join(OUT_DIR, "global_ratio_hist.png"))
        plt.close()
    if not df_pr.empty:
        agg = (
            df_pr.groupby("threshold")[["precision", "recall", "f1"]]
            .mean()
            .reset_index()
        )
        plt.figure(figsize=(6, 4))
        plt.plot(agg["threshold"], agg["precision"], label="Precision")
        plt.plot(agg["threshold"], agg["recall"], label="Recall")
        plt.plot(agg["threshold"], agg["f1"], label="F1")
        plt.xlabel("Threshold")
        plt.ylabel("Score")
        plt.title("Global PR/F1")
        plt.legend()
        plt.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(OUT_DIR, "global_pr_f1.png"))
        plt.close()

    print(f"✅ Done! Results saved to {OUT_DIR}")


if __name__ == "__main__":
    main()
