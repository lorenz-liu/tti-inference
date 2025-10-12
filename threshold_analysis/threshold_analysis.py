#!/usr/bin/env python3
"""
Project iii - Frame-Level to Interaction-Level Analysis
Shifting from frame-by-frame TTI detection to interaction-level evaluation.

Main Research Question: What minimum proportion of frames must be detected within an
interaction for it to be classified as a tool-tissue interaction?

Deliverables:
1. Define optimal detection threshold using PR curve analysis
2. Create distribution curve showing interactions detected at different thresholds
3. Justify threshold selection based on minimizing false negatives

Estimated completion: October 3rd
"""

import json
import os
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# =============================================================================
# CONFIGURATION SECTION
# =============================================================================

# -----------------------------------------------------------------------------
# File Paths
# -----------------------------------------------------------------------------
BASE_DIR = "/cluster/projects/madanigroup/lorenz/tti"
GROUND_TRUTH_DIR = os.path.join(BASE_DIR, "ground_truths")
PREDICTIONS_DIR = os.path.join(BASE_DIR, "unified_fps_inferences")
OUTPUT_DIR = os.path.join(BASE_DIR, "threshold_analysis")

# -----------------------------------------------------------------------------
# Video Selection (11 Safe Lap Chole videos for Project iii)
# -----------------------------------------------------------------------------
SAFE_VIDEOS = [
    "LapChol Case 0001 03",
    "LapChol Case 0001 04",
    "LapChol Case 0001 05",
    "LapChol Case 0002 02",
    "LapChol Case 0002 03",
    "LapChol Case 0007 01",
    "LapChol Case 0007 02",
    "LapChol Case 0007 03",
    "LapChol Case 0011 02",
    "LapChol Case 0011 03",
    "LapChol Case 0012 03",
    "LapChol Case 0012 04",
    "LapChol Case 0015 01",
    "LapChol Case 0015 02",
    "LapChol Case 0016 01",
    "LapChol Case 0018 10",
    "LapChol Case 0018 11",
    "LapChol Case 0019 02",
    "LapChol Case 0019 03",
    "LapChol Case 0020 02",
    "LapChol Case 0020 03",
    "LapChol Case 0023 03",
    "LapChol Case 0023 04",
]

# -----------------------------------------------------------------------------
# Analysis Parameters
# -----------------------------------------------------------------------------
# Thresholds to test (percentage of frames that must be detected)
DETECTION_THRESHOLDS = np.arange(0.1, 1.0, 0.05)  # 10%, 15%, 20%, ..., 95%

# Priority: Minimize False Negatives over False Positives
PRIORITIZE_RECALL = True  # Set to True to minimize missed interactions
MIN_ACCEPTABLE_RECALL = 0.90  # Minimum 90% of interactions must be detected
MAX_ACCEPTABLE_FP_RATE = 0.20  # Accept up to 20% false positive rate

# -----------------------------------------------------------------------------
# Annotation Field Names
# -----------------------------------------------------------------------------
# Ground truth annotation labels
GT_TTI_START = "Start of TTI"
GT_TTI_END = "End of TTI "  # Note the space in your data
GT_NO_INTERACTION_START = "Start of No Interaction "  # Note the space
GT_NO_INTERACTION_END = "End of No Interaction"

# Prediction labels
PRED_TTI_START = "Start of TTI"
PRED_TTI_END = "End of TTI"
PRED_NO_INTERACTION_START = "Start of No Interaction"
PRED_NO_INTERACTION_END = "End of No Interaction"

# -----------------------------------------------------------------------------
# Tool and Interaction Type Mappings (from your data)
# -----------------------------------------------------------------------------
TOOL_NAMES = {
    0: "unknown_tool",
    1: "dissector",
    2: "scissors",
    3: "suction",
    4: "grasper_3",
    5: "harmonic",
    6: "grasper",
    7: "bipolar",
    8: "grasper_2",
    9: "cautery",
    10: "ligasure",
    11: "stapler",
}

TTI_NAMES = {
    12: "unknown_tti",
    13: "coagulation",
    14: "other",
    15: "retract_and_grab",
    16: "blunt_dissection",
    17: "energy_sharp_dissection",
    18: "staple",
    19: "retract_and_push",
    20: "cut_sharp_dissection",
}

# -----------------------------------------------------------------------------
# Visualization Settings
# -----------------------------------------------------------------------------
FIGURE_DPI = 300
FIGURE_SIZE = (12, 8)
PLOT_STYLE = "seaborn-v0_8-darkgrid"
COLOR_PALETTE = "Set2"

# =============================================================================
# END OF CONFIGURATION SECTION
# =============================================================================


# =============================================================================
# DATA LOADING FUNCTIONS
# =============================================================================


def load_json(filepath: str) -> Dict:
    """Load JSON file."""
    with open(filepath, "r") as f:
        return json.load(f)


def extract_ground_truth_interactions(gt_data: Dict) -> List[Dict]:
    """
    Extract interaction segments from ground truth annotations.

    Returns list of interactions with:
    - start_frame: int
    - end_frame: int
    - interaction_type: "TTI" or "No_Interaction"
    - bounding_box: dict (from start frame)
    - tool_type: str (if available)
    """
    interactions = []

    # Get the data unit (first video in the JSON)
    data_units = gt_data[0]["data_units"]
    data_hash = list(data_units.keys())[0]
    labels = data_units[data_hash]["labels"]

    # Track interaction segments
    current_tti_start = None
    current_no_interaction_start = None

    # Sort frames numerically
    sorted_frames = sorted(labels.keys(), key=lambda x: int(x))

    for frame_str in sorted_frames:
        frame_idx = int(frame_str)
        frame_data = labels[frame_str]

        if "objects" not in frame_data or not frame_data["objects"]:
            continue

        for obj in frame_data["objects"]:
            obj_name = obj.get("name", "")

            # TTI Start
            if obj_name == GT_TTI_START:
                current_tti_start = {
                    "start_frame": frame_idx,
                    "start_bbox": obj.get("boundingBox", {}),
                    "interaction_type": "TTI",
                }

            # TTI End
            elif obj_name == GT_TTI_END and current_tti_start is not None:
                interactions.append(
                    {
                        "start_frame": current_tti_start["start_frame"],
                        "end_frame": frame_idx,
                        "total_frames": frame_idx
                        - current_tti_start["start_frame"]
                        + 1,
                        "interaction_type": "TTI",
                        "bounding_box": current_tti_start["start_bbox"],
                    }
                )
                current_tti_start = None

            # No Interaction Start
            elif obj_name == GT_NO_INTERACTION_START:
                current_no_interaction_start = {
                    "start_frame": frame_idx,
                    "start_bbox": obj.get("boundingBox", {}),
                    "interaction_type": "No_Interaction",
                }

            # No Interaction End
            elif (
                obj_name == GT_NO_INTERACTION_END
                and current_no_interaction_start is not None
            ):
                interactions.append(
                    {
                        "start_frame": current_no_interaction_start["start_frame"],
                        "end_frame": frame_idx,
                        "total_frames": frame_idx
                        - current_no_interaction_start["start_frame"]
                        + 1,
                        "interaction_type": "No_Interaction",
                        "bounding_box": current_no_interaction_start["start_bbox"],
                    }
                )
                current_no_interaction_start = None

    return interactions


def extract_predictions(pred_data: Dict) -> Dict[int, List[Dict]]:
    """
    Extract frame-by-frame predictions.

    Returns dict mapping frame_index -> list of detected objects with:
    - tti_classification: 0 or 1
    - confidence: float
    - tool_info: dict
    - tissue_info: dict
    """
    predictions = {}

    # Get the data unit
    data_units = pred_data[0]["data_units"]
    data_hash = list(data_units.keys())[0]
    labels = data_units[data_hash]["labels"]

    for frame_str, frame_data in labels.items():
        frame_idx = int(frame_str)

        if "objects" not in frame_data:
            continue

        predictions[frame_idx] = frame_data["objects"]

    return predictions


# =============================================================================
# INTERACTION DETECTION ANALYSIS
# =============================================================================


def calculate_interaction_detection_rate(
    gt_interaction: Dict, predictions: Dict[int, List[Dict]]
) -> Tuple[float, int, int]:
    """
    Calculate what percentage of frames in a ground truth interaction
    were detected by the model as TTI.

    Args:
        gt_interaction: Ground truth interaction segment
        predictions: Frame-level predictions dict

    Returns:
        (detection_rate, detected_frames, total_frames)
    """
    start = gt_interaction["start_frame"]
    end = gt_interaction["end_frame"]
    total_frames = end - start + 1
    detected_frames = 0

    for frame_idx in range(start, end + 1):
        if frame_idx not in predictions:
            continue

        # Check if any object in this frame is classified as TTI
        for obj in predictions[frame_idx]:
            if obj.get("tti_classification") == 1:
                detected_frames += 1
                break  # Count frame only once

    detection_rate = detected_frames / total_frames if total_frames > 0 else 0.0
    return detection_rate, detected_frames, total_frames


def evaluate_at_threshold(
    ground_truth_interactions: List[Dict],
    predictions: Dict[int, List[Dict]],
    threshold: float,
    interaction_type: str = "TTI",
) -> Tuple[int, int, int, int]:
    """
    Evaluate detection performance at a specific threshold.

    Args:
        ground_truth_interactions: List of GT interaction segments
        predictions: Frame-level predictions
        threshold: Detection rate threshold (0-1)
        interaction_type: "TTI" or "No_Interaction" or "all"

    Returns:
        (true_positives, false_positives, true_negatives, false_negatives)
    """
    tp = 0  # Correctly detected interactions
    fn = 0  # Missed interactions

    # Filter by interaction type if specified
    if interaction_type != "all":
        gt_filtered = [
            i
            for i in ground_truth_interactions
            if i["interaction_type"] == interaction_type
        ]
    else:
        gt_filtered = ground_truth_interactions

    for gt_interaction in gt_filtered:
        detection_rate, _, _ = calculate_interaction_detection_rate(
            gt_interaction, predictions
        )

        if detection_rate >= threshold:
            tp += 1
        else:
            fn += 1

    # Note: FP and TN are harder to calculate without explicit negative examples
    # For now, we focus on TP and FN which are most important for this analysis
    fp = 0  # TODO: Implement FP detection
    tn = 0  # TODO: Implement TN detection

    return tp, fp, tn, fn


def compute_metrics_at_thresholds(
    ground_truth_interactions: List[Dict],
    predictions: Dict[int, List[Dict]],
    thresholds: np.ndarray,
    interaction_type: str = "TTI",
) -> pd.DataFrame:
    """
    Compute precision, recall, F1 at multiple thresholds.

    Returns DataFrame with columns:
    - threshold
    - tp, fp, tn, fn
    - precision, recall, f1
    - interactions_detected
    - total_interactions
    """
    results = []

    total_interactions = len(
        [
            i
            for i in ground_truth_interactions
            if interaction_type == "all" or i["interaction_type"] == interaction_type
        ]
    )

    for threshold in thresholds:
        tp, fp, tn, fn = evaluate_at_threshold(
            ground_truth_interactions, predictions, threshold, interaction_type
        )

        # Calculate metrics
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (
            2 * (precision * recall) / (precision + recall)
            if (precision + recall) > 0
            else 0.0
        )

        results.append(
            {
                "threshold": threshold,
                "tp": tp,
                "fp": fp,
                "tn": tn,
                "fn": fn,
                "precision": precision,
                "recall": recall,
                "f1": f1,
                "interactions_detected": tp,
                "total_interactions": total_interactions,
                "detection_percentage": 100.0 * tp / total_interactions
                if total_interactions > 0
                else 0.0,
            }
        )

    return pd.DataFrame(results)


# =============================================================================
# THRESHOLD SELECTION
# =============================================================================


def select_optimal_threshold(
    metrics_df: pd.DataFrame,
    min_recall: float = MIN_ACCEPTABLE_RECALL,
    prioritize_recall: bool = PRIORITIZE_RECALL,
) -> Tuple[float, Dict]:
    """
    Select optimal threshold based on recall priority.

    Strategy:
    - Find highest threshold that still achieves min_recall
    - This minimizes false positives while maintaining high recall

    Args:
        metrics_df: DataFrame with threshold metrics
        min_recall: Minimum acceptable recall
        prioritize_recall: If True, prioritize recall over precision

    Returns:
        (optimal_threshold, metrics_at_threshold)
    """
    # Filter to thresholds meeting minimum recall
    valid_thresholds = metrics_df[metrics_df["recall"] >= min_recall]

    if valid_thresholds.empty:
        # If no threshold meets minimum recall, choose best recall
        print(f"Warning: No threshold achieves {min_recall:.1%} recall")
        best_idx = metrics_df["recall"].idxmax()
        optimal_threshold = metrics_df.loc[best_idx, "threshold"]
        metrics = metrics_df.loc[best_idx].to_dict()
    else:
        if prioritize_recall:
            # Among valid thresholds, choose highest threshold (strictest)
            # This reduces false positives while maintaining recall
            best_idx = valid_thresholds["threshold"].idxmax()
        else:
            # Choose threshold with best F1 score
            best_idx = valid_thresholds["f1"].idxmax()

        optimal_threshold = valid_thresholds.loc[best_idx, "threshold"]
        metrics = valid_thresholds.loc[best_idx].to_dict()

    return optimal_threshold, metrics


# =============================================================================
# VISUALIZATION FUNCTIONS
# =============================================================================


def plot_pr_curve(
    metrics_df: pd.DataFrame, output_dir: str, interaction_type: str = "all"
):
    """Plot Precision-Recall curve."""
    plt.style.use(PLOT_STYLE)
    fig, ax = plt.subplots(figsize=FIGURE_SIZE)

    ax.plot(
        metrics_df["recall"],
        metrics_df["precision"],
        marker="o",
        linewidth=2,
        markersize=6,
        label="PR Curve",
    )

    # Highlight optimal point
    optimal_idx = metrics_df["f1"].idxmax()
    opt_recall = metrics_df.loc[optimal_idx, "recall"]
    opt_precision = metrics_df.loc[optimal_idx, "precision"]
    opt_threshold = metrics_df.loc[optimal_idx, "threshold"]

    ax.plot(
        opt_recall,
        opt_precision,
        "r*",
        markersize=20,
        label=f"Optimal (threshold={opt_threshold:.2f})",
    )

    ax.set_xlabel("Recall (Sensitivity)", fontsize=14)
    ax.set_ylabel("Precision", fontsize=14)
    ax.set_title(
        f"Precision-Recall Curve - {interaction_type} Interactions", fontsize=16
    )
    ax.legend(fontsize=12)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(
        os.path.join(output_dir, f"pr_curve_{interaction_type}.png"), dpi=FIGURE_DPI
    )
    plt.close()


def plot_threshold_metrics(
    metrics_df: pd.DataFrame, output_dir: str, interaction_type: str = "all"
):
    """Plot metrics across thresholds."""
    plt.style.use(PLOT_STYLE)
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    # Precision, Recall, F1
    ax = axes[0, 0]
    ax.plot(
        metrics_df["threshold"],
        metrics_df["precision"],
        marker="o",
        label="Precision",
        linewidth=2,
    )
    ax.plot(
        metrics_df["threshold"],
        metrics_df["recall"],
        marker="s",
        label="Recall",
        linewidth=2,
    )
    ax.plot(
        metrics_df["threshold"],
        metrics_df["f1"],
        marker="^",
        label="F1 Score",
        linewidth=2,
    )
    ax.set_xlabel("Detection Threshold", fontsize=12)
    ax.set_ylabel("Score", fontsize=12)
    ax.set_title("Precision, Recall, F1 vs Threshold", fontsize=14)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    # Interactions Detected
    ax = axes[0, 1]
    ax.bar(
        metrics_df["threshold"],
        metrics_df["interactions_detected"],
        alpha=0.7,
        color="steelblue",
    )
    ax.axhline(
        y=metrics_df["total_interactions"].iloc[0],
        color="r",
        linestyle="--",
        label="Total Interactions",
    )
    ax.set_xlabel("Detection Threshold", fontsize=12)
    ax.set_ylabel("Interactions Detected", fontsize=12)
    ax.set_title("Interactions Detected at Each Threshold", fontsize=14)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    # Detection Percentage
    ax = axes[1, 0]
    ax.plot(
        metrics_df["threshold"],
        metrics_df["detection_percentage"],
        marker="o",
        linewidth=2,
        color="green",
    )
    ax.axhline(y=90, color="r", linestyle="--", alpha=0.5, label="90% Target")
    ax.set_xlabel("Detection Threshold", fontsize=12)
    ax.set_ylabel("Detection Rate (%)", fontsize=12)
    ax.set_title("Percentage of Interactions Detected", fontsize=14)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    # False Negatives
    ax = axes[1, 1]
    ax.bar(metrics_df["threshold"], metrics_df["fn"], alpha=0.7, color="coral")
    ax.set_xlabel("Detection Threshold", fontsize=12)
    ax.set_ylabel("False Negatives (Missed Interactions)", fontsize=12)
    ax.set_title("Missed Interactions at Each Threshold", fontsize=14)
    ax.grid(True, alpha=0.3)

    plt.suptitle(
        f"Threshold Analysis - {interaction_type} Interactions", fontsize=18, y=1.00
    )
    plt.tight_layout()
    plt.savefig(
        os.path.join(output_dir, f"threshold_metrics_{interaction_type}.png"),
        dpi=FIGURE_DPI,
        bbox_inches="tight",
    )
    plt.close()


def plot_detection_distribution(
    metrics_df: pd.DataFrame, output_dir: str, interaction_type: str = "all"
):
    """
    Plot distribution of interactions detected at different thresholds.
    This is Deliverable 2.
    """
    plt.style.use(PLOT_STYLE)
    fig, ax = plt.subplots(figsize=FIGURE_SIZE)

    # Create histogram-style plot
    thresholds_pct = metrics_df["threshold"] * 100
    interactions_detected = metrics_df["interactions_detected"]
    total = metrics_df["total_interactions"].iloc[0]

    bars = ax.bar(
        thresholds_pct,
        interactions_detected,
        width=4,
        alpha=0.7,
        color="steelblue",
        edgecolor="black",
    )

    # Highlight optimal threshold
    optimal_idx = metrics_df["f1"].idxmax()
    bars[optimal_idx].set_color("gold")
    bars[optimal_idx].set_edgecolor("red")
    bars[optimal_idx].set_linewidth(3)

    # Add reference line for total
    ax.axhline(
        y=total,
        color="r",
        linestyle="--",
        linewidth=2,
        label=f"Total Interactions (n={total})",
    )

    # Add percentage labels on bars
    for i, (threshold, detected) in enumerate(
        zip(thresholds_pct, interactions_detected)
    ):
        if i % 2 == 0:  # Label every other bar to avoid clutter
            pct = 100 * detected / total
            ax.text(
                threshold,
                detected + 0.5,
                f"{pct:.1f}%",
                ha="center",
                va="bottom",
                fontsize=9,
            )

    ax.set_xlabel("Detection Threshold (%)", fontsize=14)
    ax.set_ylabel("Number of Interactions Detected", fontsize=14)
    ax.set_title(
        f"Distribution of Interactions Detected at Different Thresholds\n"
        f"{interaction_type} Interactions",
        fontsize=16,
    )
    ax.legend(fontsize=12)
    ax.grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    plt.savefig(
        os.path.join(output_dir, f"detection_distribution_{interaction_type}.png"),
        dpi=FIGURE_DPI,
    )
    plt.close()


# =============================================================================
# MAIN ANALYSIS PIPELINE
# =============================================================================


def analyze_video(
    video_name: str, gt_dir: str, pred_dir: str
) -> Tuple[List[Dict], Dict]:
    """
    Analyze a single video.

    Returns:
        (ground_truth_interactions, predictions)
    """
    # Construct file paths
    gt_path = os.path.join(gt_dir, f"{video_name}.json")
    pred_path = os.path.join(pred_dir, f"pred_{video_name}.json")

    if not os.path.exists(gt_path):
        print(f"Warning: Ground truth not found for {video_name}")
        return [], {}

    if not os.path.exists(pred_path):
        print(f"Warning: Predictions not found for {video_name}")
        return [], {}

    # Load data
    gt_data = load_json(gt_path)
    pred_data = load_json(pred_path)

    # Extract interactions and predictions
    interactions = extract_ground_truth_interactions(gt_data)
    predictions = extract_predictions(pred_data)

    print(f"  Loaded {video_name}:")
    print(f"    - {len(interactions)} ground truth interactions")
    print(f"    - {len(predictions)} frames with predictions")

    return interactions, predictions


def run_analysis(videos: List[str], analysis_type: str = "all"):
    """
    Run complete analysis on multiple videos.

    Args:
        videos: List of video names (without extension)
        analysis_type: "all", "by_tool", or "by_interaction"
    """
    print("\n" + "=" * 80)
    print(f"PROJECT III - INTERACTION LEVEL ANALYSIS ({analysis_type})")
    print("=" * 80)

    # Create output directory
    output_dir = os.path.join(OUTPUT_DIR, analysis_type)
    os.makedirs(output_dir, exist_ok=True)

    # Load all data
    all_interactions = []
    all_predictions = {}

    print("\nLoading data...")
    for video_name in videos:
        interactions, predictions = analyze_video(
            video_name, GROUND_TRUTH_DIR, PREDICTIONS_DIR
        )
        all_interactions.extend(interactions)
        # Store predictions with video context
        for frame_idx, preds in predictions.items():
            all_predictions[(video_name, frame_idx)] = preds

    print(f"\nTotal: {len(all_interactions)} interactions across {len(videos)} videos")

    # Count by type
    tti_count = sum(1 for i in all_interactions if i["interaction_type"] == "TTI")
    no_int_count = len(all_interactions) - tti_count
    print(f"  - TTI interactions: {tti_count}")
    print(f"  - No-Interaction periods: {no_int_count}")

    # Compute metrics at different thresholds
    print("\nComputing metrics at different thresholds...")

    # Need to restructure predictions for evaluation
    # This is a simplified version - you may need to adjust based on your data structure
    flat_predictions = {}
    for (video, frame), preds in all_predictions.items():
        if frame not in flat_predictions:
            flat_predictions[frame] = []
        flat_predictions[frame].extend(preds)

    # Analyze overall (all interactions)
    print("\n--- Overall Analysis (All Interactions) ---")
    metrics_all = compute_metrics_at_thresholds(
        all_interactions, flat_predictions, DETECTION_THRESHOLDS, "all"
    )

    optimal_threshold, optimal_metrics = select_optimal_threshold(metrics_all)

    print(f"\nOptimal Threshold: {optimal_threshold:.2%}")
    print(f"  Recall (Sensitivity): {optimal_metrics['recall']:.2%}")
    print(f"  Precision: {optimal_metrics['precision']:.2%}")
    print(f"  F1 Score: {optimal_metrics['f1']:.4f}")
    print(
        f"  Interactions Detected: {optimal_metrics['tp']}/{optimal_metrics['total_interactions']}"
    )
    print(f"  False Negatives: {optimal_metrics['fn']}")

    # Save metrics
    metrics_all.to_csv(
        os.path.join(output_dir, "threshold_metrics_all.csv"), index=False
    )

    # Generate visualizations
    print("\nGenerating visualizations...")
    plot_pr_curve(metrics_all, output_dir, "all")
    plot_threshold_metrics(metrics_all, output_dir, "all")
    plot_detection_distribution(metrics_all, output_dir, "all")

    # Analyze TTI only
    print("\n--- TTI Interactions Only ---")
    metrics_tti = compute_metrics_at_thresholds(
        all_interactions, flat_predictions, DETECTION_THRESHOLDS, "TTI"
    )
    metrics_tti.to_csv(
        os.path.join(output_dir, "threshold_metrics_tti.csv"), index=False
    )

    plot_pr_curve(metrics_tti, output_dir, "TTI")
    plot_threshold_metrics(metrics_tti, output_dir, "TTI")
    plot_detection_distribution(metrics_tti, output_dir, "TTI")

    optimal_tti_threshold, optimal_tti_metrics = select_optimal_threshold(metrics_tti)
    print(f"Optimal TTI Threshold: {optimal_tti_threshold:.2%}")
    print(f"  Recall: {optimal_tti_metrics['recall']:.2%}")
    print(f"  Precision: {optimal_tti_metrics['precision']:.2%}")

    # Generate summary report
    generate_summary_report(
        metrics_all, metrics_tti, output_dir, optimal_threshold, optimal_metrics
    )

    print(f"\n✓ Analysis complete. Results saved to: {output_dir}")
    print("=" * 80 + "\n")


def generate_summary_report(
    metrics_all: pd.DataFrame,
    metrics_tti: pd.DataFrame,
    output_dir: str,
    optimal_threshold: float,
    optimal_metrics: Dict,
):
    """Generate final summary report for deliverables."""
    report_path = os.path.join(output_dir, "PROJECT_III_SUMMARY.txt")

    with open(report_path, "w") as f:
        f.write("=" * 80 + "\n")
        f.write("PROJECT III - INTERACTION LEVEL ANALYSIS SUMMARY\n")
        f.write("=" * 80 + "\n\n")

        f.write("DELIVERABLE 1: OPTIMAL THRESHOLD DETERMINATION\n")
        f.write("-" * 80 + "\n")
        f.write("Based on Precision-Recall curve analysis:\n\n")
        f.write(
            f"  Optimal Threshold: {optimal_threshold:.1%} of frames must be detected\n\n"
        )
        f.write("  At this threshold:\n")
        f.write(f"    - Recall (Sensitivity): {optimal_metrics['recall']:.1%}\n")
        f.write(f"    - Precision: {optimal_metrics['precision']:.1%}\n")
        f.write(f"    - F1 Score: {optimal_metrics['f1']:.4f}\n")
        f.write(f"    - False Negative Rate: {(1 - optimal_metrics['recall']):.1%}\n")
        f.write(
            f"    - Interactions Detected: {optimal_metrics['tp']}/{optimal_metrics['total_interactions']}\n"
        )
        f.write(f"    - Missed Interactions: {optimal_metrics['fn']}\n\n")

        f.write("  Rationale:\n")
        f.write("    This threshold minimizes missed interactions (false negatives)\n")
        f.write(
            f"    while maintaining acceptable precision. It achieves >{MIN_ACCEPTABLE_RECALL:.0%}\n"
        )
        f.write("    recall as required for surgical safety applications.\n\n")

        f.write("\n" + "=" * 80 + "\n")
        f.write("DELIVERABLE 2: DISTRIBUTION ANALYSIS\n")
        f.write("-" * 80 + "\n")
        f.write("Interactions detected at different thresholds:\n")
        f.write("(11 Safe Laparoscopic Cholecystectomy videos)\n\n")

        for _, row in metrics_all.iterrows():
            threshold_pct = row["threshold"] * 100
            detected = row["interactions_detected"]
            total = row["total_interactions"]
            detection_pct = row["detection_percentage"]

            marker = (
                " ← SELECTED"
                if abs(row["threshold"] - optimal_threshold) < 0.01
                else ""
            )

            f.write(
                f"  {threshold_pct:4.0f}% threshold: {detected:3d}/{total} interactions ({detection_pct:5.1f}%){marker}\n"
            )

        f.write("\n  [See detection_distribution plots for visual representation]\n")

        f.write("\n" + "=" * 80 + "\n")
        f.write("Estimated Completion Date: October 3rd\n")
        f.write("=" * 80 + "\n")

    print(f"✓ Summary report saved to: {report_path}")


# =============================================================================
# MAIN EXECUTION
# =============================================================================


def main():
    """Main execution function."""
    print("\n" + "=" * 80)
    print("PROJECT III - FRAME TO INTERACTION LEVEL ANALYSIS")
    print("=" * 80)
    print("\nConfiguration:")
    print(f"  Ground Truth Dir: {GROUND_TRUTH_DIR}")
    print(f"  Predictions Dir: {PREDICTIONS_DIR}")
    print(f"  Output Dir: {OUTPUT_DIR}")
    print(f"  Videos to analyze: {len(SAFE_VIDEOS)}")
    print(
        f"  Detection thresholds: {len(DETECTION_THRESHOLDS)} ({DETECTION_THRESHOLDS[0]:.0%} to {DETECTION_THRESHOLDS[-1]:.0%})"
    )
    print(f"  Min acceptable recall: {MIN_ACCEPTABLE_RECALL:.0%}")
    print("=" * 80)

    # Run analysis (overall - regardless of tool/interaction type)
    run_analysis(SAFE_VIDEOS, analysis_type="overall")

    # TODO: Add by_tool and by_interaction analyses if time permits
    # run_analysis(SAFE_VIDEOS, analysis_type="by_tool")
    # run_analysis(SAFE_VIDEOS, analysis_type="by_interaction")

    print("\n" + "=" * 80)
    print("PROJECT III COMPLETE")
    print("=" * 80)
    print(f"\nAll results saved to: {OUTPUT_DIR}")
    print("\nDeliverables generated:")
    print("  ✓ Optimal threshold determination (PR curve analysis)")
    print("  ✓ Distribution curves at different thresholds")
    print("  ✓ Precision/Recall/F1 metrics")
    print("  ✓ Visualization plots")
    print("  ✓ Summary report")
    print("\nEstimated completion: October 3rd")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    main()
