#!/usr/bin/env python3
"""
Calculate IoU and DICE scores between ground truth and predicted TTI bounding boxes.

This script:
1. Loads ground truth JSON files
2. Finds frames with exactly one "start_of_tti" label
3. Extracts those frames from videos
4. Runs inference on each frame using optimized_eval.py
5. Compares predicted tti_bounding_box with ground truth boundingBox
6. Calculates IoU and DICE scores
7. Generates visualizations and saves results

Output:
- combined_results.json: All results combined
- <video_name>_results.json: Per-video detailed results
- <video_name>_metrics.png: Per-video IoU and DICE plots
- <video_name>_frame_<N>_gt.jpg: Frame with ground truth bounding box (green)
- <video_name>_frame_<N>_pred.jpg: Frame with prediction bounding box (blue)
- overall_summary.png: Summary plot across all videos

Usage:
    # Use all defaults (cluster paths)
    python calculate_iou_dice.py

    # Specify custom paths
    python calculate_iou_dice.py --ground_truth_dir /path/to/ground_truths --video_dir /path/to/videos --output /path/to/output_dir
"""

import argparse
import json
import os
import subprocess
import tempfile
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import matplotlib.pyplot as plt
import numpy as np

# ============================================================================
# CONFIGURATION CONSTANTS
# ============================================================================

# Default model paths
DEFAULT_VIT_MODEL_PATH = "/cluster/projects/madanigroup/lorenz/tti/vit.pt"
DEFAULT_YOLO_MODEL_PATH = "/cluster/projects/madanigroup/lorenz/tti/yolo.pt"

# Inference settings
INFERENCE_SCRIPT = "/cluster/projects/madanigroup/lorenz/tti/optimized_eval.py"
INFERENCE_TIMEOUT = 6000  # 1 Hour

# Video settings
DEFAULT_VIDEO_FPS = 30.0

# TTI label identifiers
TTI_LABEL_VALUES = ["start_of_tti"]
TTI_LABEL_NAMES = ["Start of TTI"]

# Hardcoded mapping between ground truth JSON filenames and video filenames
# This handles the naming inconsistencies between JSON and video files
GT_TO_VIDEO_MAPPING = {
    # LapChol cases - exact matches
    "LapChol Case 0001 03.mp4": "LapChol Case 0001 03.mp4",
    "LapChol Case 0001 04.mp4": "LapChol Case 0001 04.mp4",
    "LapChol Case 0001 05.mp4": "LapChol Case 0001 05.mp4",
    "LapChol Case 0002 02.mp4": "LapChol Case 0002 02.mp4",
    "LapChol Case 0002 03.mp4": "LapChol Case 0002 03.mp4",
    "LapChol Case 0007 01.mp4": "LapChol Case 0007 01.mp4",
    "LapChol Case 0007 02.mp4": "LapChol Case 0007 02.mp4",
    "LapChol Case 0007 03.mp4": "LapChol Case 0007 03.mp4",
    "LapChol Case 0011 02.mp4": "LapChol Case 0011 02.mp4",
    "LapChol Case 0011 03.mp4": "LapChol Case 0011 03.mp4",
    "LapChol Case 0012 03.mp4": "LapChol Case 0012 03.mp4",
    "LapChol Case 0012 04.mp4": "LapChol Case 0012 04.mp4",
    "LapChol Case 0015 01.mp4": "LapChol Case 0015 01.mp4",
    "LapChol Case 0015 02.mp4": "LapChol Case 0015 02.mp4",
    "LapChol Case 0016 01.mp4": "LapChol Case 0016 01.mp4",
    "LapChol Case 0018 10.mp4": "LapChol Case 0018 10.mp4",
    "LapChol Case 0018 11.mp4": "LapChol Case 0018 11.mp4",
    "LapChol Case 0019 02.mp4": "LapChol Case 0019 02.mp4",
    "LapChol Case 0019 03.mp4": "LapChol Case 0019 03.mp4",
    "LapChol Case 0020 02.mp4": "LapChol Case 0020 02.mp4",
    "LapChol Case 0020 03.mp4": "LapChol Case 0020 03.mp4",
    "LapChol Case 0023 03.mp4": "LapChol Case 0023 03.mp4",
    "LapChol Case 0023 04.mp4": "LapChol Case 0023 04.mp4",
    # V cases - naming inconsistencies (space vs dash vs underscore)
    "V10 Trimmed.mp4": "V10-Trimmed.mp4",
    "V11 Trimmed.mp4": "V11-Trimmed.mp4",
    "V12 Trimmed.mp4": "V12-Trimmed.mp4",
    "V14 Trimmed.mp4": "V14_Trimmed.mp4",
    "V15 Trimmed.mp4": "V15_Trimmed.mp4",
    "V17 Trimmed.mp4": "V17_Trimmed.mp4",
    "V18 Trimmed.mp4": "V18_Trimmed.mp4",
    "V2 Trimmed.mp4": "V2_Trimmed.mp4",
    "V4 Trimmed.mp4": "V4_Trimmed.mp4",
    "V5 Trimmed.mp4": "V5_Trimmed.mp4",
    "V7 Trimmed.mp4": "V7-Trimmed.mp4",
}

# Absolute paths (set these based on your environment)
DEFAULT_GROUND_TRUTH_DIR = "/cluster/projects/madanigroup/lorenz/tti/ground_truths"
DEFAULT_VIDEO_DIR = "/cluster/projects/madanigroup/lorenz/tti/videos"
DEFAULT_OUTPUT_DIR = "/cluster/projects/madanigroup/lorenz/tti/unified_fps_iou_dice"


def calculate_iou(box1: Dict[str, float], box2: Dict[str, float]) -> float:
    """
    Calculate Intersection over Union (IoU) between two bounding boxes.

    Args:
        box1: First bounding box with keys 'x', 'y', 'w', 'h' (normalized coordinates)
        box2: Second bounding box with keys 'x', 'y', 'w', 'h' (normalized coordinates)

    Returns:
        IoU score (0.0 to 1.0)
    """
    # Convert to (x1, y1, x2, y2) format
    x1_min = box1["x"]
    y1_min = box1["y"]
    x1_max = box1["x"] + box1["w"]
    y1_max = box1["y"] + box1["h"]

    x2_min = box2["x"]
    y2_min = box2["y"]
    x2_max = box2["x"] + box2["w"]
    y2_max = box2["y"] + box2["h"]

    # Calculate intersection area
    x_inter_min = max(x1_min, x2_min)
    y_inter_min = max(y1_min, y2_min)
    x_inter_max = min(x1_max, x2_max)
    y_inter_max = min(y1_max, y2_max)

    if x_inter_max < x_inter_min or y_inter_max < y_inter_min:
        return 0.0

    intersection = (x_inter_max - x_inter_min) * (y_inter_max - y_inter_min)

    # Calculate union area
    area1 = box1["w"] * box1["h"]
    area2 = box2["w"] * box2["h"]
    union = area1 + area2 - intersection

    if union == 0:
        return 0.0

    return intersection / union


def calculate_dice(box1: Dict[str, float], box2: Dict[str, float]) -> float:
    """
    Calculate DICE coefficient between two bounding boxes.

    Args:
        box1: First bounding box with keys 'x', 'y', 'w', 'h' (normalized coordinates)
        box2: Second bounding box with keys 'x', 'y', 'w', 'h' (normalized coordinates)

    Returns:
        DICE score (0.0 to 1.0)
    """
    # Convert to (x1, y1, x2, y2) format
    x1_min = box1["x"]
    y1_min = box1["y"]
    x1_max = box1["x"] + box1["w"]
    y1_max = box1["y"] + box1["h"]

    x2_min = box2["x"]
    y2_min = box2["y"]
    x2_max = box2["x"] + box2["w"]
    y2_max = box2["y"] + box2["h"]

    # Calculate intersection area
    x_inter_min = max(x1_min, x2_min)
    y_inter_min = max(y1_min, y2_min)
    x_inter_max = min(x1_max, x2_max)
    y_inter_max = min(y1_max, y2_max)

    if x_inter_max < x_inter_min or y_inter_max < y_inter_min:
        return 0.0

    intersection = (x_inter_max - x_inter_min) * (y_inter_max - y_inter_min)

    # Calculate areas
    area1 = box1["w"] * box1["h"]
    area2 = box2["w"] * box2["h"]

    if area1 + area2 == 0:
        return 0.0

    # DICE = 2 * |A ∩ B| / (|A| + |B|)
    return (2 * intersection) / (area1 + area2)


def load_ground_truth(gt_path: str) -> Dict:
    """Load ground truth JSON file."""
    with open(gt_path, "r") as f:
        return json.load(f)


def find_single_tti_frames(gt_data: Dict) -> List[Tuple[int, Dict]]:
    """
    Find frames with exactly one "start_of_tti" label.

    Args:
        gt_data: Ground truth data structure

    Returns:
        List of tuples: (frame_index, bounding_box_dict)
    """
    single_tti_frames = []

    # Navigate to labels
    if isinstance(gt_data, list):
        gt_data = gt_data[0]

    for data_unit_key, data_unit_value in gt_data["data_units"].items():
        labels = data_unit_value.get("labels", {})

        for frame_key, frame_data in labels.items():
            frame_idx = int(frame_key)
            objects = frame_data.get("objects", [])

            # Find all start_of_tti objects
            tti_objects = [
                obj
                for obj in objects
                if obj.get("value") in TTI_LABEL_VALUES
                or obj.get("name") in TTI_LABEL_NAMES
            ]

            # Only include frames with exactly one TTI
            if len(tti_objects) == 1:
                bbox = tti_objects[0]["boundingBox"]
                single_tti_frames.append((frame_idx, bbox))

    return single_tti_frames


def extract_frame_from_video(video_path: str, frame_idx: int, output_path: str) -> bool:
    """
    Extract a specific frame from a video and save as an image.

    Args:
        video_path: Path to video file
        frame_idx: Frame index to extract
        output_path: Path to save extracted frame

    Returns:
        True if successful, False otherwise
    """
    try:
        cap = cv2.VideoCapture(video_path)
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        cap.release()

        if ret:
            cv2.imwrite(output_path, frame)
            return True
        return False
    except Exception as e:
        print(f"Error extracting frame {frame_idx} from {video_path}: {e}")
        return False


def create_single_frame_video(
    frame_path: str, output_video_path: str, fps: float = DEFAULT_VIDEO_FPS
) -> bool:
    """
    Create a single-frame video from an image.

    Args:
        frame_path: Path to input frame image
        output_video_path: Path to output video
        fps: Frames per second for the video

    Returns:
        True if successful, False otherwise
    """
    try:
        frame = cv2.imread(frame_path)
        if frame is None:
            return False

        height, width = frame.shape[:2]
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))
        out.write(frame)
        out.release()
        return True
    except Exception as e:
        print(f"Error creating single-frame video: {e}")
        return False


def run_inference_on_frame(
    video_path: str,
    output_json_path: str,
    vit_model: str = DEFAULT_VIT_MODEL_PATH,
    yolo_model: str = DEFAULT_YOLO_MODEL_PATH,
) -> Optional[Dict]:
    """
    Run optimized_eval.py on a single-frame video.

    Args:
        video_path: Path to single-frame video
        output_json_path: Path to save prediction JSON
        vit_model: Path to ViT model
        yolo_model: Path to YOLO model

    Returns:
        Prediction data dict or None if failed
    """
    try:
        cmd = [
            "python",
            INFERENCE_SCRIPT,
            "--video",
            video_path,
            "--output",
            output_json_path,
            "--vit_model",
            vit_model,
            "--yolo_model",
            yolo_model,
            "--start_frame",
            "0",
            "--end_frame",
            "1",
            "--frame_step",
            "1",
        ]

        result = subprocess.run(
            cmd, capture_output=True, text=True, timeout=INFERENCE_TIMEOUT
        )

        if result.returncode != 0:
            print(f"Inference failed with return code {result.returncode}")
            print(f"stderr: {result.stderr}")
            return None

        # Load prediction results
        with open(output_json_path, "r") as f:
            return json.load(f)

    except subprocess.TimeoutExpired:
        print(f"Inference timed out for {video_path}")
        return None
    except Exception as e:
        print(f"Error running inference: {e}")
        return None


def extract_prediction_tti_bboxes(pred_data: Dict, frame_idx: int = 0) -> List[Dict]:
    """
    Extract all tti_bounding_boxes from prediction data that have start_of_tti.

    Args:
        pred_data: Prediction data structure
        frame_idx: Frame index to check

    Returns:
        List of tti_bounding_box dicts with additional metadata
    """
    if isinstance(pred_data, list):
        pred_data = pred_data[0]

    tti_bboxes = []

    for data_unit_key, data_unit_value in pred_data["data_units"].items():
        labels = data_unit_value.get("labels", {})
        frame_data = labels.get(str(frame_idx))

        if not frame_data:
            return []

        objects = frame_data.get("objects", [])

        # Find all start_of_tti objects with tti_classification == 1
        tti_objects = [
            obj
            for obj in objects
            if obj.get("tti_classification") == 1
            and (
                obj.get("value") in TTI_LABEL_VALUES
                or obj.get("name") in TTI_LABEL_NAMES
            )
        ]

        # Return all TTI bounding boxes with metadata
        for obj in tti_objects:
            tti_bbox = obj.get("tti_bounding_box")
            if tti_bbox:
                tti_bboxes.append(
                    {
                        "tti_bounding_box": tti_bbox,
                        "confidence": obj.get("confidence"),
                        "tool_info": obj.get("tool_info"),
                        "tissue_info": obj.get("tissue_info"),
                        "full_object": obj,
                    }
                )

    return tti_bboxes


def draw_bbox_on_frame(
    frame: np.ndarray,
    bbox: Dict[str, float],
    color: tuple = (0, 255, 0),
    thickness: int = 3,
    label: str = None,
) -> np.ndarray:
    """
    Draw a bounding box on a frame.

    Args:
        frame: Input frame (BGR format)
        bbox: Bounding box dict with keys 'x', 'y', 'w', 'h' (normalized coordinates)
        color: BGR color tuple for the box
        thickness: Line thickness
        label: Optional label text to draw

    Returns:
        Frame with bounding box drawn
    """
    frame_with_bbox = frame.copy()
    h, w = frame.shape[:2]

    # Convert normalized coordinates to pixel coordinates
    x = int(bbox["x"] * w)
    y = int(bbox["y"] * h)
    box_w = int(bbox["w"] * w)
    box_h = int(bbox["h"] * h)

    # Draw rectangle
    cv2.rectangle(frame_with_bbox, (x, y), (x + box_w, y + box_h), color, thickness)

    # Draw label if provided
    if label:
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.8
        font_thickness = 2

        # Get text size for background
        (text_w, text_h), _ = cv2.getTextSize(label, font, font_scale, font_thickness)

        # Draw background rectangle for text
        text_bg_y = y - text_h - 10 if y > text_h + 10 else y + box_h + 5
        cv2.rectangle(
            frame_with_bbox,
            (x, text_bg_y - 5),
            (x + text_w + 10, text_bg_y + text_h + 5),
            color,
            -1,
        )

        # Draw text
        cv2.putText(
            frame_with_bbox,
            label,
            (x + 5, text_bg_y + text_h),
            font,
            font_scale,
            (255, 255, 255),
            font_thickness,
        )

    return frame_with_bbox


def save_annotated_frames(
    frame: np.ndarray,
    gt_bbox: Dict[str, float],
    pred_bbox: Dict[str, float],
    output_dir: str,
    video_name: str,
    frame_idx: int,
    iou: float,
    dice: float,
):
    """
    Save frames with ground truth and prediction bounding boxes.

    Args:
        frame: Input frame (BGR format)
        gt_bbox: Ground truth bounding box
        pred_bbox: Predicted bounding box
        output_dir: Directory to save images
        video_name: Video filename
        frame_idx: Frame index
        iou: IoU score
        dice: DICE score
    """
    safe_video_name = video_name.replace(" ", "_").replace(".mp4", "")

    # Draw ground truth bbox (green)
    gt_label = "GT"
    frame_gt = draw_bbox_on_frame(
        frame, gt_bbox, color=(0, 255, 0), thickness=3, label=gt_label
    )

    # Draw prediction bbox (blue)
    pred_label = f"Pred (IoU:{iou:.3f} DICE:{dice:.3f})"
    frame_pred = draw_bbox_on_frame(
        frame, pred_bbox, color=(255, 0, 0), thickness=3, label=pred_label
    )

    # Save ground truth frame
    gt_filename = f"{safe_video_name}_frame_{frame_idx}_gt.jpg"
    gt_path = os.path.join(output_dir, gt_filename)
    cv2.imwrite(gt_path, frame_gt)

    # Save prediction frame
    pred_filename = f"{safe_video_name}_frame_{frame_idx}_pred.jpg"
    pred_path = os.path.join(output_dir, pred_filename)
    cv2.imwrite(pred_path, frame_pred)

    print(f"      Saved annotated frames: {gt_filename}, {pred_filename}")


def process_ground_truth_file(
    gt_path: str,
    video_dir: str,
    temp_dir: str,
    vit_model: str,
    yolo_model: str,
    output_dir: str,
) -> List[Dict]:
    """
    Process a single ground truth file and calculate metrics.

    Args:
        gt_path: Path to ground truth JSON file
        video_dir: Directory containing video files
        temp_dir: Temporary directory for intermediate files
        vit_model: Path to ViT model
        yolo_model: Path to YOLO model
        output_dir: Directory to save annotated frames

    Returns:
        List of result dictionaries with metrics
    """
    results = []

    print(f"\nProcessing ground truth: {gt_path}")
    gt_data = load_ground_truth(gt_path)

    # Get video filename from ground truth
    if isinstance(gt_data, list):
        gt_video_filename = gt_data[0]["data_title"]
    else:
        gt_video_filename = gt_data["data_title"]

    # Use mapping to get actual video filename
    actual_video_filename = GT_TO_VIDEO_MAPPING.get(
        gt_video_filename, gt_video_filename
    )

    if gt_video_filename != actual_video_filename:
        print(f"Mapped GT filename '{gt_video_filename}' -> '{actual_video_filename}'")

    video_path = os.path.join(video_dir, actual_video_filename)

    if not os.path.exists(video_path):
        print(f"Warning: Video not found: {video_path}")
        print(f"  GT filename: {gt_video_filename}")
        print(f"  Mapped filename: {actual_video_filename}")
        return results

    print(f"Video: {actual_video_filename}")

    # Find frames with single TTI
    single_tti_frames = find_single_tti_frames(gt_data)
    print(f"Found {len(single_tti_frames)} frames with exactly one TTI annotation")

    for frame_idx, gt_bbox in single_tti_frames:
        print(f"\n  Processing frame {frame_idx}...")

        # Create temp paths
        frame_img_path = os.path.join(temp_dir, f"frame_{frame_idx}.png")
        frame_video_path = os.path.join(temp_dir, f"frame_{frame_idx}.mp4")
        pred_json_path = os.path.join(temp_dir, f"pred_{frame_idx}.json")

        # Extract frame
        if not extract_frame_from_video(video_path, frame_idx, frame_img_path):
            print(f"    Failed to extract frame {frame_idx}")
            continue

        # Create single-frame video
        if not create_single_frame_video(frame_img_path, frame_video_path):
            print(f"    Failed to create single-frame video for frame {frame_idx}")
            continue

        # Run inference
        pred_data = run_inference_on_frame(
            frame_video_path, pred_json_path, vit_model, yolo_model
        )

        if pred_data is None:
            print(f"    Inference failed for frame {frame_idx}")
            results.append(
                {
                    "ground_truth_file": os.path.basename(gt_path),
                    "video_file": actual_video_filename,
                    "gt_video_filename": gt_video_filename,
                    "frame": frame_idx,
                    "status": "inference_failed",
                    "iou": None,
                    "dice": None,
                }
            )
            continue

        # Extract all prediction bboxes with TTI
        pred_tti_bboxes = extract_prediction_tti_bboxes(pred_data, frame_idx=0)

        if not pred_tti_bboxes:
            print(f"    No TTI detected in prediction for frame {frame_idx}")
            results.append(
                {
                    "ground_truth_file": os.path.basename(gt_path),
                    "video_file": actual_video_filename,
                    "gt_video_filename": gt_video_filename,
                    "frame": frame_idx,
                    "status": "no_tti_predicted",
                    "iou": 0.0,
                    "dice": 0.0,
                    "ground_truth_bbox": gt_bbox,
                    "num_predictions": 0,
                }
            )
            continue

        # Calculate metrics for all predicted TTI bboxes
        print(
            f"    Found {len(pred_tti_bboxes)} TTI prediction(s), calculating metrics..."
        )

        best_match = None
        best_iou = -1
        all_matches = []

        for idx, pred_info in enumerate(pred_tti_bboxes):
            pred_bbox = pred_info["tti_bounding_box"]
            iou = calculate_iou(gt_bbox, pred_bbox)
            dice = calculate_dice(gt_bbox, pred_bbox)

            match_info = {
                "prediction_index": idx,
                "iou": iou,
                "dice": dice,
                "predicted_bbox": pred_bbox,
                "confidence": pred_info["confidence"],
                "tool_info": pred_info["tool_info"],
                "tissue_info": pred_info["tissue_info"],
            }
            all_matches.append(match_info)

            print(
                f"      Prediction {idx}: IoU={iou:.4f}, DICE={dice:.4f}, Conf={pred_info['confidence']:.4f}"
            )

            # Track best match by IoU
            if iou > best_iou:
                best_iou = iou
                best_match = match_info

        # Use the best match
        print(
            f"    Best match: Prediction {best_match['prediction_index']} with IoU={best_match['iou']:.4f}, DICE={best_match['dice']:.4f}"
        )

        # Load the frame for annotation
        frame_for_annotation = cv2.imread(frame_img_path)
        if frame_for_annotation is not None:
            # Save annotated frames with bounding boxes
            save_annotated_frames(
                frame_for_annotation,
                gt_bbox,
                best_match["predicted_bbox"],
                output_dir,
                actual_video_filename,
                frame_idx,
                best_match["iou"],
                best_match["dice"],
            )

        results.append(
            {
                "ground_truth_file": os.path.basename(gt_path),
                "video_file": actual_video_filename,
                "gt_video_filename": gt_video_filename,
                "frame": frame_idx,
                "status": "success",
                "iou": best_match["iou"],
                "dice": best_match["dice"],
                "ground_truth_bbox": gt_bbox,
                "predicted_bbox": best_match["predicted_bbox"],
                "num_predictions": len(pred_tti_bboxes),
                "best_prediction_index": best_match["prediction_index"],
                "best_prediction_confidence": best_match["confidence"],
                "best_prediction_tool": best_match["tool_info"],
                "best_prediction_tissue": best_match["tissue_info"],
                "all_predictions": all_matches,
            }
        )

        # Clean up temp files
        for temp_file in [frame_img_path, frame_video_path, pred_json_path]:
            if os.path.exists(temp_file):
                os.remove(temp_file)

    return results


def create_per_video_plots(video_results: List[Dict], output_dir: str, video_name: str):
    """
    Create matplotlib plots for a single video's IoU and DICE scores.

    Args:
        video_results: List of result dicts for a single video
        output_dir: Directory to save plots
        video_name: Name of the video (for filename)
    """
    successful_results = [r for r in video_results if r["status"] == "success"]

    if not successful_results:
        print(f"  No successful results for {video_name}, skipping plots")
        return

    # Extract frame indices and metrics
    frames = [r["frame"] for r in successful_results]
    ious = [r["iou"] for r in successful_results]
    dices = [r["dice"] for r in successful_results]

    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
    fig.suptitle(f"IoU and DICE Scores - {video_name}", fontsize=16, fontweight="bold")

    # Plot IoU
    ax1.plot(frames, ious, "b-o", linewidth=2, markersize=6, label="IoU per frame")
    ax1.axhline(
        y=np.mean(ious),
        color="r",
        linestyle="--",
        linewidth=2,
        label=f"Mean IoU: {np.mean(ious):.4f}",
    )
    ax1.fill_between(frames, ious, alpha=0.3)
    ax1.set_xlabel("Frame Index", fontsize=12)
    ax1.set_ylabel("IoU Score", fontsize=12)
    ax1.set_title("Intersection over Union (IoU)", fontsize=14)
    ax1.grid(True, alpha=0.3)
    ax1.legend(fontsize=10)
    ax1.set_ylim([0, 1])

    # Plot DICE
    ax2.plot(frames, dices, "g-o", linewidth=2, markersize=6, label="DICE per frame")
    ax2.axhline(
        y=np.mean(dices),
        color="r",
        linestyle="--",
        linewidth=2,
        label=f"Mean DICE: {np.mean(dices):.4f}",
    )
    ax2.fill_between(frames, dices, alpha=0.3)
    ax2.set_xlabel("Frame Index", fontsize=12)
    ax2.set_ylabel("DICE Score", fontsize=12)
    ax2.set_title("DICE Coefficient", fontsize=14)
    ax2.grid(True, alpha=0.3)
    ax2.legend(fontsize=10)
    ax2.set_ylim([0, 1])

    plt.tight_layout()

    # Save plot
    safe_video_name = video_name.replace(" ", "_").replace(".mp4", "")
    plot_path = os.path.join(output_dir, f"{safe_video_name}_metrics.png")
    plt.savefig(plot_path, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"  Saved plot: {plot_path}")


def create_overall_summary_plot(all_results: List[Dict], output_dir: str):
    """
    Create overall summary plots showing average IoU and DICE across all videos.

    Args:
        all_results: List of all result dicts
        output_dir: Directory to save plots
    """
    # Group results by video
    video_metrics = {}
    for result in all_results:
        if result["status"] == "success":
            video_name = result["video_file"]
            if video_name not in video_metrics:
                video_metrics[video_name] = {"ious": [], "dices": []}
            video_metrics[video_name]["ious"].append(result["iou"])
            video_metrics[video_name]["dices"].append(result["dice"])

    if not video_metrics:
        print("No successful results to plot")
        return

    # Calculate mean metrics per video
    video_names = []
    mean_ious = []
    mean_dices = []

    for video_name in sorted(video_metrics.keys()):
        video_names.append(video_name.replace(".mp4", "").replace(" ", "\n"))
        mean_ious.append(np.mean(video_metrics[video_name]["ious"]))
        mean_dices.append(np.mean(video_metrics[video_name]["dices"]))

    # Create figure
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 12))
    fig.suptitle(
        "Overall Summary - Mean IoU and DICE per Video", fontsize=16, fontweight="bold"
    )

    x_pos = np.arange(len(video_names))

    # Plot mean IoU per video
    bars1 = ax1.bar(x_pos, mean_ious, alpha=0.7, color="blue", edgecolor="black")
    ax1.axhline(
        y=np.mean(mean_ious),
        color="r",
        linestyle="--",
        linewidth=2,
        label=f"Overall Mean IoU: {np.mean(mean_ious):.4f}",
    )
    ax1.set_ylabel("Mean IoU Score", fontsize=12)
    ax1.set_title("Mean IoU per Video", fontsize=14)
    ax1.set_xticks(x_pos)
    ax1.set_xticklabels(video_names, rotation=45, ha="right", fontsize=8)
    ax1.grid(True, alpha=0.3, axis="y")
    ax1.legend(fontsize=10)
    ax1.set_ylim([0, 1])

    # Add value labels on bars
    for bar in bars1:
        height = bar.get_height()
        ax1.text(
            bar.get_x() + bar.get_width() / 2.0,
            height,
            f"{height:.3f}",
            ha="center",
            va="bottom",
            fontsize=7,
        )

    # Plot mean DICE per video
    bars2 = ax2.bar(x_pos, mean_dices, alpha=0.7, color="green", edgecolor="black")
    ax2.axhline(
        y=np.mean(mean_dices),
        color="r",
        linestyle="--",
        linewidth=2,
        label=f"Overall Mean DICE: {np.mean(mean_dices):.4f}",
    )
    ax2.set_ylabel("Mean DICE Score", fontsize=12)
    ax2.set_title("Mean DICE per Video", fontsize=14)
    ax2.set_xticks(x_pos)
    ax2.set_xticklabels(video_names, rotation=45, ha="right", fontsize=8)
    ax2.grid(True, alpha=0.3, axis="y")
    ax2.legend(fontsize=10)
    ax2.set_ylim([0, 1])

    # Add value labels on bars
    for bar in bars2:
        height = bar.get_height()
        ax2.text(
            bar.get_x() + bar.get_width() / 2.0,
            height,
            f"{height:.3f}",
            ha="center",
            va="bottom",
            fontsize=7,
        )

    plt.tight_layout()

    # Save plot
    plot_path = os.path.join(output_dir, "overall_summary.png")
    plt.savefig(plot_path, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"\nSaved overall summary plot: {plot_path}")


def save_per_video_results(all_results: List[Dict], output_dir: str):
    """
    Save individual JSON files for each video's results.

    Args:
        all_results: List of all result dicts
        output_dir: Directory to save JSON files
    """
    # Group results by video
    video_results = {}
    for result in all_results:
        video_name = result["video_file"]
        if video_name not in video_results:
            video_results[video_name] = []
        video_results[video_name].append(result)

    # Save each video's results
    for video_name, results in video_results.items():
        # Calculate statistics for this video
        successful = [r for r in results if r["status"] == "success"]

        video_summary = {
            "video_file": video_name,
            "total_frames": len(results),
            "successful_comparisons": len(successful),
            "failed_comparisons": len(results) - len(successful),
        }

        if successful:
            ious = [r["iou"] for r in successful]
            dices = [r["dice"] for r in successful]

            video_summary.update(
                {
                    "mean_iou": float(np.mean(ious)),
                    "std_iou": float(np.std(ious)),
                    "median_iou": float(np.median(ious)),
                    "min_iou": float(np.min(ious)),
                    "max_iou": float(np.max(ious)),
                    "mean_dice": float(np.mean(dices)),
                    "std_dice": float(np.std(dices)),
                    "median_dice": float(np.median(dices)),
                    "min_dice": float(np.min(dices)),
                    "max_dice": float(np.max(dices)),
                }
            )

        # Create output structure
        output_data = {"summary": video_summary, "detailed_results": results}

        # Save to JSON
        safe_video_name = video_name.replace(" ", "_").replace(".mp4", "")
        json_path = os.path.join(output_dir, f"{safe_video_name}_results.json")
        with open(json_path, "w") as f:
            json.dump(output_data, f, indent=2)

        print(f"  Saved results: {json_path}")

        # Create plots for this video
        create_per_video_plots(results, output_dir, video_name)


def calculate_summary_statistics(results: List[Dict]) -> Dict:
    """Calculate summary statistics from results."""
    successful_results = [r for r in results if r["status"] == "success"]

    if not successful_results:
        return {
            "total_frames": len(results),
            "successful_comparisons": 0,
            "failed_comparisons": len(results),
            "mean_iou": None,
            "mean_dice": None,
            "median_iou": None,
            "median_dice": None,
            "std_iou": None,
            "std_dice": None,
        }

    ious = [r["iou"] for r in successful_results]
    dices = [r["dice"] for r in successful_results]

    # Calculate statistics about multiple predictions
    num_predictions_list = [r.get("num_predictions", 1) for r in successful_results]
    frames_with_multiple_predictions = sum(1 for n in num_predictions_list if n > 1)

    return {
        "total_frames": len(results),
        "successful_comparisons": len(successful_results),
        "failed_comparisons": len(results) - len(successful_results),
        "mean_iou": float(np.mean(ious)),
        "mean_dice": float(np.mean(dices)),
        "median_iou": float(np.median(ious)),
        "median_dice": float(np.median(dices)),
        "std_iou": float(np.std(ious)),
        "std_dice": float(np.std(dices)),
        "min_iou": float(np.min(ious)),
        "max_iou": float(np.max(ious)),
        "min_dice": float(np.min(dices)),
        "max_dice": float(np.max(dices)),
        "frames_with_multiple_predictions": frames_with_multiple_predictions,
        "mean_predictions_per_frame": float(np.mean(num_predictions_list)),
        "max_predictions_in_frame": int(np.max(num_predictions_list)),
    }


def main():
    parser = argparse.ArgumentParser(
        description="Calculate IoU and DICE scores between ground truth and predicted TTI bounding boxes"
    )
    parser.add_argument(
        "--ground_truth_dir",
        default=DEFAULT_GROUND_TRUTH_DIR,
        help=f"Directory containing ground truth JSON files (default: {DEFAULT_GROUND_TRUTH_DIR})",
    )
    parser.add_argument(
        "--video_dir",
        default=DEFAULT_VIDEO_DIR,
        help=f"Directory containing video files (default: {DEFAULT_VIDEO_DIR})",
    )
    parser.add_argument(
        "--output",
        default=DEFAULT_OUTPUT_DIR,
        help=f"Directory to save all results (default: {DEFAULT_OUTPUT_DIR})",
    )
    parser.add_argument(
        "--vit_model", default=DEFAULT_VIT_MODEL_PATH, help="Path to ViT model"
    )
    parser.add_argument(
        "--yolo_model", default=DEFAULT_YOLO_MODEL_PATH, help="Path to YOLO model"
    )
    parser.add_argument(
        "--temp_dir",
        default=None,
        help="Temporary directory for intermediate files (default: system temp)",
    )

    args = parser.parse_args()

    # Validate inputs
    if not os.path.exists(args.ground_truth_dir):
        print(f"Error: Ground truth directory not found: {args.ground_truth_dir}")
        return

    if not os.path.exists(args.video_dir):
        print(f"Error: Video directory not found: {args.video_dir}")
        return

    # Create output directory
    output_dir = args.output
    os.makedirs(output_dir, exist_ok=True)
    print(f"Output directory: {output_dir}")

    # Setup temp directory
    if args.temp_dir:
        temp_dir = args.temp_dir
        os.makedirs(temp_dir, exist_ok=True)
    else:
        temp_dir = tempfile.mkdtemp()

    print(f"Using temporary directory: {temp_dir}")

    # Find all ground truth JSON files
    gt_files = sorted(Path(args.ground_truth_dir).glob("*.json"))
    print(f"\nFound {len(gt_files)} ground truth files")

    # Process each ground truth file
    all_results = []
    for gt_path in gt_files:
        results = process_ground_truth_file(
            str(gt_path),
            args.video_dir,
            temp_dir,
            args.vit_model,
            args.yolo_model,
            output_dir,
        )
        all_results.extend(results)

    # Calculate summary statistics
    summary = calculate_summary_statistics(all_results)

    # Save overall results
    print("\n" + "=" * 60)
    print("SAVING RESULTS")
    print("=" * 60)

    # Save combined results JSON
    combined_output_data = {"summary": summary, "detailed_results": all_results}
    combined_json_path = os.path.join(output_dir, "combined_results.json")
    with open(combined_json_path, "w") as f:
        json.dump(combined_output_data, f, indent=2)
    print(f"Saved combined results: {combined_json_path}")

    # Save per-video results and plots
    print("\nSaving per-video results and plots...")
    save_per_video_results(all_results, output_dir)

    # Create overall summary plot
    print("\nCreating overall summary plot...")
    create_overall_summary_plot(all_results, output_dir)

    # Print summary statistics
    print("\n" + "=" * 60)
    print("SUMMARY STATISTICS")
    print("=" * 60)
    print(f"Total frames processed: {summary['total_frames']}")
    print(f"Successful comparisons: {summary['successful_comparisons']}")
    print(f"Failed comparisons: {summary['failed_comparisons']}")

    if summary["mean_iou"] is not None:
        print(f"\nMean IoU: {summary['mean_iou']:.4f} ± {summary['std_iou']:.4f}")
        print(f"Median IoU: {summary['median_iou']:.4f}")
        print(f"Range IoU: [{summary['min_iou']:.4f}, {summary['max_iou']:.4f}]")

        print(f"\nMean DICE: {summary['mean_dice']:.4f} ± {summary['std_dice']:.4f}")
        print(f"Median DICE: {summary['median_dice']:.4f}")
        print(f"Range DICE: [{summary['min_dice']:.4f}, {summary['max_dice']:.4f}]")

        print("\nMultiple Predictions Statistics:")
        print(
            f"Frames with multiple predictions: {summary['frames_with_multiple_predictions']}"
        )
        print(
            f"Mean predictions per frame: {summary['mean_predictions_per_frame']:.2f}"
        )
        print(f"Max predictions in a frame: {summary['max_predictions_in_frame']}")

    print(f"\nAll results saved to: {output_dir}")
    print("=" * 60)

    # Clean up temp directory if we created it
    if not args.temp_dir:
        import shutil

        shutil.rmtree(temp_dir)


if __name__ == "__main__":
    main()

