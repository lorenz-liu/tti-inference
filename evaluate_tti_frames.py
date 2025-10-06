#!/usr/bin/env python3
"""
TTI Frame-Level Evaluation Script

Evaluates TTI detection performance by comparing ground truth annotations with model predictions
on frames where TTI starts or ends. Computes IoU and DICE scores for bounding box overlap.

Usage:
  python evaluate_tti_frames.py --annotations_dir /path/to/annotations --videos_dir /path/to/videos --output /path/to/results.json [--vit_model PATH] [--yolo_model PATH] [--device cuda|cpu|mps] [--batch_size N] [--workers N]

Arguments:
- --annotations_dir: Directory containing ground truth JSON annotation files. Required.
- --videos_dir: Directory containing video files. Required.
- --output: Path to save evaluation results JSON. Required.
- --vit_model: Path to ViT classifier weights. Default: ./model_folder/ViT/best_model.pt
- --yolo_model: Path to YOLO segmentation weights. Default: auto-detect from project paths
- --device: Compute device (cuda/cpu/mps). Default: auto-detect
- --batch_size: Batch size for processing frames. Default: 16
- --workers: Number of parallel workers for frame extraction. Default: 4

Output:
- Detailed per-frame IoU and DICE scores
- Overall statistics (mean, std, min, max)
- Confusion matrix for TTI detection
"""

import argparse
import glob
import json
import os
import warnings
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from tqdm import tqdm
from transformers import pipeline

warnings.filterwarnings("ignore")

# Import local modules
from ultralytics import YOLO

# ============================================================================
# Configuration Constants
# ============================================================================

DEFAULT_VIDEO_PATH = "../videos/"
DEFAULT_GROUND_TRUTH_PATH = "../ground_truth_annos/"
DEFAULT_VIT_MODEL_PATH = "../model_folder/ViT/best_model.pt"
DEFAULT_YOLO_MODEL_PATH = "../runs_YOLO_M/segment/train/weights/best.pt"
YOLO_PROJECT_MODEL_PATHS = [
    "../runs_YOLO_M/segment/train/weights/best.pt",
    "../runs_YOLOn_200/segment/train/weights/best.pt",
    "../runs_yolon_new/segment/train/weights/best.pt",
    "../runs/segment/train/weights/best.pt",
]

DEFAULT_DEPTH_MODEL_ID = "Intel/dpt-large"
DEFAULT_BATCH_SIZE = 16
DEFAULT_WORKERS = 4
DEFAULT_FOCUS_RATIO = 1.0

# ViT architecture config
VIT_IMAGE_SIZE = 224
VIT_PATCH_SIZE = 16
VIT_NUM_CHANNELS = 3
VIT_HIDDEN_SIZE = 768
VIT_NUM_LAYERS = 12
VIT_NUM_HEADS = 12
VIT_INTERMEDIATE_SIZE = 3072
VIT_HIDDEN_ACT = "gelu"
VIT_HIDDEN_DROPOUT = 0.0
VIT_ATTENTION_DROPOUT = 0.0
VIT_INIT_RANGE = 0.02
VIT_LAYER_NORM_EPS = 1e-12
VIT_QKV_BIAS = True

# Class mappings
TOOL_CLASSES = list(range(0, 12))
TTI_CLASSES = list(range(12, 21))

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

ROI_MIN_SIZE = 64
DEPTH_CACHE_MAX_SIZE = 100


# ============================================================================
# Utility Functions
# ============================================================================


def compute_iou(box1: Dict, box2: Dict) -> float:
    """
    Compute IoU between two bounding boxes in normalized coordinates.

    Args:
        box1, box2: Dicts with keys 'x', 'y', 'w', 'h' (normalized 0-1)

    Returns:
        IoU score (0-1)
    """
    x1_min = box1["x"]
    y1_min = box1["y"]
    x1_max = box1["x"] + box1["w"]
    y1_max = box1["y"] + box1["h"]

    x2_min = box2["x"]
    y2_min = box2["y"]
    x2_max = box2["x"] + box2["w"]
    y2_max = box2["y"] + box2["h"]

    # Intersection
    x_inter_min = max(x1_min, x2_min)
    y_inter_min = max(y1_min, y2_min)
    x_inter_max = min(x1_max, x2_max)
    y_inter_max = min(y1_max, y2_max)

    if x_inter_max <= x_inter_min or y_inter_max <= y_inter_min:
        return 0.0

    inter_area = (x_inter_max - x_inter_min) * (y_inter_max - y_inter_min)

    # Union
    box1_area = box1["w"] * box1["h"]
    box2_area = box2["w"] * box2["h"]
    union_area = box1_area + box2_area - inter_area

    if union_area == 0:
        return 0.0

    return inter_area / union_area


def compute_dice(box1: Dict, box2: Dict) -> float:
    """
    Compute DICE coefficient between two bounding boxes.

    Args:
        box1, box2: Dicts with keys 'x', 'y', 'w', 'h' (normalized 0-1)

    Returns:
        DICE score (0-1)
    """
    x1_min = box1["x"]
    y1_min = box1["y"]
    x1_max = box1["x"] + box1["w"]
    y1_max = box1["y"] + box1["h"]

    x2_min = box2["x"]
    y2_min = box2["y"]
    x2_max = box2["x"] + box2["w"]
    y2_max = box2["y"] + box2["h"]

    # Intersection
    x_inter_min = max(x1_min, x2_min)
    y_inter_min = max(y1_min, y2_min)
    x_inter_max = min(x1_max, x2_max)
    y_inter_max = min(y1_max, y2_max)

    if x_inter_max <= x_inter_min or y_inter_max <= y_inter_min:
        return 0.0

    inter_area = (x_inter_max - x_inter_min) * (y_inter_max - y_inter_min)

    # DICE = 2 * intersection / (area1 + area2)
    box1_area = box1["w"] * box1["h"]
    box2_area = box2["w"] * box2["h"]

    if box1_area + box2_area == 0:
        return 0.0

    return 2 * inter_area / (box1_area + box2_area)


def extract_frame(video_path: str, frame_idx: int) -> Optional[np.ndarray]:
    """Extract a specific frame from video."""
    cap = cv2.VideoCapture(video_path)
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
    ret, frame = cap.read()
    cap.release()

    if not ret:
        return None
    return frame


# ============================================================================
# TTI Evaluator (adapted from optimized_eval.py)
# ============================================================================


class TTIFrameEvaluator:
    """TTI evaluator for single frame inference"""

    def __init__(
        self,
        vit_model_path,
        yolo_model_path=None,
        device=None,
        focus_ratio=1.0,
        depth_model_path=None,
        use_half_precision=True,
    ):
        """Initialize the TTI frame evaluator"""

        # Device selection
        if device:
            self.device = device
        else:
            if torch.cuda.is_available():
                self.device = "cuda"
            elif torch.backends.mps.is_available():
                self.device = "mps"
            else:
                self.device = "cpu"

        self.focus_ratio = focus_ratio

        # Bind class mappings
        self.TOOL_CLASSES = TOOL_CLASSES
        self.TTI_CLASSES = TTI_CLASSES
        self.TOOL_NAMES = TOOL_NAMES
        self.TTI_NAMES = TTI_NAMES

        # Half precision settings
        if self.device == "cuda":
            self.use_half_precision_vit = use_half_precision
            self.use_half_precision_yolo = False
        else:
            self.use_half_precision_vit = False
            self.use_half_precision_yolo = False

        # Enable optimizations
        if self.device == "cuda":
            torch.backends.cudnn.benchmark = True
            torch.backends.cudnn.deterministic = False

        # Load models
        self.tti_model = self._load_tti_model(vit_model_path)
        self.yolo_model = self._load_yolo_model(yolo_model_path)
        self._load_depth_model(depth_model_path)

        # Cache for depth maps
        self.depth_cache = {}

    def _load_tti_model(self, model_path):
        """Load the trained ViT TTI classifier"""
        import torch.nn as nn
        from transformers import ViTConfig, ViTModel

        class OptimizedROIClassifierViT(nn.Module):
            def __init__(self, num_hoi_classes):
                super().__init__()
                self.first = nn.Conv2d(
                    5, 4, kernel_size=1, stride=1, padding=0, bias=False
                )
                self.pre_conv = nn.Conv2d(
                    4, 3, kernel_size=1, stride=1, padding=0, bias=False
                )

                config = ViTConfig(
                    image_size=VIT_IMAGE_SIZE,
                    patch_size=VIT_PATCH_SIZE,
                    num_channels=VIT_NUM_CHANNELS,
                    hidden_size=VIT_HIDDEN_SIZE,
                    num_hidden_layers=VIT_NUM_LAYERS,
                    num_attention_heads=VIT_NUM_HEADS,
                    intermediate_size=VIT_INTERMEDIATE_SIZE,
                    hidden_act=VIT_HIDDEN_ACT,
                    hidden_dropout_prob=VIT_HIDDEN_DROPOUT,
                    attention_probs_dropout_prob=VIT_ATTENTION_DROPOUT,
                    initializer_range=VIT_INIT_RANGE,
                    layer_norm_eps=VIT_LAYER_NORM_EPS,
                    qkv_bias=VIT_QKV_BIAS,
                )

                self.backbone = ViTModel(config)
                self.fc = nn.Linear(768, num_hoi_classes)

            def forward(self, x):
                x = self.first(x)
                x = self.pre_conv(x)
                outputs = self.backbone(pixel_values=x)
                features = outputs.last_hidden_state[:, 0]
                out = F.sigmoid(self.fc(features))
                return out

        model = OptimizedROIClassifierViT(num_hoi_classes=2)
        checkpoint = torch.load(model_path, map_location=self.device)
        model.load_state_dict(checkpoint)
        model.to(self.device)

        if self.use_half_precision_vit:
            model.half()

        model.eval()
        return model

    def _load_yolo_model(self, model_path):
        """Load YOLO model"""
        if model_path and os.path.exists(model_path):
            model = YOLO(model_path)
        else:
            for path in YOLO_PROJECT_MODEL_PATHS:
                if os.path.exists(path):
                    model = YOLO(path)
                    break
            else:
                raise ValueError("No YOLO model found")

        if self.device == "cuda":
            try:
                model.model.float()
            except Exception:
                pass

        return model

    def _load_depth_model(self, depth_model_path):
        """Load depth estimation model"""
        try:
            if depth_model_path and os.path.exists(depth_model_path):
                device_id = 0 if self.device in ["cuda", "mps"] else -1
                self.depth_model = pipeline(
                    task="depth-estimation", model=depth_model_path, device=device_id
                )
            else:
                device_id = 0 if self.device in ["cuda", "mps"] else -1
                self.depth_model = pipeline(
                    task="depth-estimation",
                    model=DEFAULT_DEPTH_MODEL_ID,
                    device=device_id,
                )
            self.use_real_depth = True
        except Exception:
            self.depth_model = None
            self.use_real_depth = False

    def _fallback_depth_estimation(self, image):
        """Simple fallback depth estimation"""
        img_array = np.array(image)
        if len(img_array.shape) == 3:
            gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
        else:
            gray = img_array

        edges = cv2.Canny(gray, 50, 150)
        dist_transform = cv2.distanceTransform(255 - edges, cv2.DIST_L2, 5)
        depth_map = 255 - cv2.normalize(
            dist_transform, None, 0, 255, cv2.NORM_MINMAX
        ).astype(np.uint8)

        return depth_map

    def parse_yolo_output(self, result):
        """Parse YOLO output to separate tools and tissues"""
        if not result or len(result) == 0:
            return []

        r = result[0]
        if r.boxes is None or r.masks is None or len(r.boxes.cls) == 0:
            return []

        classes = r.boxes.cls.cpu().numpy()
        masks = r.masks.data.cpu().numpy()
        boxes = r.boxes.data.cpu().numpy()
        confidences = r.boxes.conf.cpu().numpy()

        tool_indices = [
            i for i, cls in enumerate(classes) if int(cls) in self.TOOL_CLASSES
        ]
        tti_indices = [
            i for i, cls in enumerate(classes) if int(cls) in self.TTI_CLASSES
        ]

        detections = []
        for idx in tool_indices + tti_indices:
            detection = {
                "class": int(classes[idx]),
                "mask": masks[idx],
                "box": boxes[idx],
                "confidence": float(confidences[idx]),
                "type": "tool" if int(classes[idx]) in self.TOOL_CLASSES else "tti",
            }
            detections.append(detection)

        return detections

    def find_tool_tissue_pairs_fast(self, detections):
        """Fast tool-tissue pair finding"""
        tools = [d for d in detections if d["type"] == "tool"]
        tissues = [d for d in detections if d["type"] == "tti"]

        pairs = []
        for tool in tools:
            for tissue in tissues:
                pairs.append({"tool": tool, "tissue": tissue})

        # Keep top 3 pairs by confidence
        if len(pairs) > 3:
            pairs = sorted(
                pairs,
                key=lambda p: p["tool"]["confidence"] + p["tissue"]["confidence"],
                reverse=True,
            )[:3]

        return pairs

    def extract_union_roi_fast(
        self, image, tool_mask, tissue_mask, depth_map, focus_ratio=1.0
    ):
        """Fast ROI extraction"""
        H, W = image.shape[:2]

        # Resize masks
        tool_mask_resized = cv2.resize(
            tool_mask.astype(np.uint8), (W, H), interpolation=cv2.INTER_NEAREST
        )
        tissue_mask_resized = cv2.resize(
            tissue_mask.astype(np.uint8), (W, H), interpolation=cv2.INTER_NEAREST
        )

        # Get union mask
        union_mask = cv2.bitwise_or(tool_mask_resized, tissue_mask_resized)

        if cv2.countNonZero(union_mask) == 0:
            return None, None

        # Get bounding box
        x, y, w, h = cv2.boundingRect(union_mask)

        # Ensure minimum size
        min_size = ROI_MIN_SIZE
        if w < min_size:
            w = min_size
        if h < min_size:
            h = min_size

        # Bounds checking
        x = max(0, min(x, W - w))
        y = max(0, min(y, H - h))
        w = min(w, W - x)
        h = min(h, H - y)

        # Extract ROI
        roi = image[y : y + h, x : x + w]
        depth_roi = depth_map[y : y + h, x : x + w]
        union_roi = union_mask[y : y + h, x : x + w]

        # Combine channels
        roi = np.concatenate([roi, depth_roi[..., None]], axis=-1)
        roi = np.concatenate([roi, union_roi[..., None]], axis=-1)

        return roi, (x, y, w, h)

    def process_frame(self, frame: np.ndarray, frame_idx: int) -> List[Dict]:
        """
        Process a single frame and return TTI predictions.

        Returns:
            List of prediction dicts with 'boundingBox' and 'tti_classification'
        """
        # YOLO inference
        try:
            yolo_results = self.yolo_model(frame, verbose=False)
        except Exception as e:
            print(f"YOLO inference failed: {e}")
            return []

        # Get depth map
        depth_cache_key = f"{frame_idx}"
        if depth_cache_key in self.depth_cache:
            depth_map = self.depth_cache[depth_cache_key]
        else:
            pil_frame = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            if self.use_real_depth and self.depth_model is not None:
                try:
                    depth_result = self.depth_model(pil_frame)
                    depth_map = np.array(depth_result["depth"])
                except Exception:
                    depth_map = self._fallback_depth_estimation(pil_frame)
            else:
                depth_map = self._fallback_depth_estimation(pil_frame)

            self.depth_cache[depth_cache_key] = depth_map
            if len(self.depth_cache) > DEPTH_CACHE_MAX_SIZE:
                oldest_key = min(self.depth_cache.keys())
                del self.depth_cache[oldest_key]

        # Parse YOLO output
        detections = self.parse_yolo_output(yolo_results)
        pairs = self.find_tool_tissue_pairs_fast(detections)

        if not pairs:
            return []

        # Collect ROIs for batch ViT inference
        roi_batch = []
        pair_info = []

        for pair in pairs:
            roi, bbox = self.extract_union_roi_fast(
                cv2.cvtColor(frame, cv2.COLOR_BGR2RGB),
                pair["tool"]["mask"],
                pair["tissue"]["mask"],
                depth_map,
                self.focus_ratio,
            )

            if roi is not None:
                roi_resized = cv2.resize(
                    roi,
                    (VIT_IMAGE_SIZE, VIT_IMAGE_SIZE),
                    interpolation=cv2.INTER_LINEAR,
                )
                roi_tensor = (
                    torch.from_numpy(roi_resized).permute(2, 0, 1).float() / 255.0
                )

                if self.use_half_precision_vit and self.device == "cuda":
                    roi_tensor = roi_tensor.half()

                roi_batch.append(roi_tensor)
                pair_info.append((pair, bbox))

        if not roi_batch:
            return []

        # Batch ViT inference
        predictions = []
        try:
            roi_batch_tensor = torch.stack(roi_batch).to(self.device)

            with torch.no_grad():
                logits_batch = self.tti_model(roi_batch_tensor)
                probabilities_batch = F.softmax(logits_batch, dim=1)
                tti_classes = torch.argmax(logits_batch, dim=1).cpu().numpy()
                tti_confidences = probabilities_batch.max(dim=1)[0].cpu().numpy()

            # Process results
            H, W = frame.shape[:2]
            for j, (pair, bbox) in enumerate(pair_info):
                tti_class = int(tti_classes[j])
                tti_confidence = float(tti_confidences[j])

                pred = {
                    "boundingBox": {
                        "x": float(bbox[0]) / W,
                        "y": float(bbox[1]) / H,
                        "w": float(bbox[2]) / W,
                        "h": float(bbox[3]) / H,
                    },
                    "tti_classification": tti_class,
                    "confidence": tti_confidence,
                    "tool_class": pair["tool"]["class"],
                    "tissue_class": pair["tissue"]["class"],
                }
                predictions.append(pred)

        except Exception as e:
            print(f"ViT inference failed: {e}")
            return []

        return predictions


# ============================================================================
# Evaluation Pipeline
# ============================================================================


def load_ground_truth_annotations(annotations_dir: str) -> Dict:
    """
    Load all ground truth annotation files.

    Returns:
        Dict mapping video filename to annotation data
    """
    annotation_files = glob.glob(os.path.join(annotations_dir, "*.json"))
    annotations_map = {}

    for ann_file in annotation_files:
        with open(ann_file, "r") as f:
            data = json.load(f)

        # Assume the annotation file name or data_title identifies the video
        if isinstance(data, list) and len(data) > 0:
            data = data[0]  # Take first element if it's a list

        video_name = data.get("data_title", "")
        if video_name:
            annotations_map[video_name] = data

    return annotations_map


def extract_tti_frames_from_annotations(annotation_data: Dict) -> List[Tuple[int, str]]:
    """
    Extract frame indices where TTI starts or ends.

    Returns:
        List of tuples (frame_idx, label_type) where label_type is 'start_of_tti' or 'end_of_tti_'
    """
    tti_frames = []

    data_units = annotation_data.get("data_units", {})
    for data_hash, data_unit in data_units.items():
        labels = data_unit.get("labels", {})

        for frame_str, frame_data in labels.items():
            frame_idx = int(frame_str)
            objects = frame_data.get("objects", [])

            for obj in objects:
                value = obj.get("value", "")
                if value in ["start_of_tti", "end_of_tti_"]:
                    tti_frames.append((frame_idx, value, obj))

    return tti_frames


def save_per_video_plot(video_name: str, video_results: List[Dict], output_dir: str):
    """
    Save IoU and DICE plots for a single video.

    Args:
        video_name: Name of the video
        video_results: List of results for this video
        output_dir: Directory to save the plot
    """
    # Use frame averages and deduplicate by frame to handle multiple ground truths per frame
    frame_data = {}
    for r in video_results:
        frame_idx = r.get("frame")
        if frame_idx is not None and "frame_avg_iou" in r and "frame_avg_dice" in r:
            # Use frame averages (all results for same frame will have same averages)
            if frame_idx not in frame_data:
                frame_data[frame_idx] = {
                    "iou": r["frame_avg_iou"],
                    "dice": r["frame_avg_dice"],
                }

    if not frame_data:
        return

    # Extract sorted frame data
    sorted_frames = sorted(frame_data.keys())
    iou_scores = [frame_data[f]["iou"] for f in sorted_frames]
    dice_scores = [frame_data[f]["dice"] for f in sorted_frames]
    frames = sorted_frames

    if not iou_scores and not dice_scores:
        return

    # Create figure with two subplots
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Sanitize video name for filename
    safe_video_name = (
        video_name.replace("/", "_").replace("\\", "_").replace(".mp4", "")
    )
    fig.suptitle(f"Evaluation Results: {video_name}", fontsize=14, fontweight="bold")

    # IoU plot
    if iou_scores:
        ax1 = axes[0]
        ax1.plot(
            frames,
            iou_scores,
            "b-o",
            linewidth=2,
            markersize=6,
            alpha=0.7,
            label="IoU scores",
        )
        ax1.axhline(
            np.mean(iou_scores),
            color="red",
            linestyle="--",
            linewidth=2,
            label=f"Mean: {np.mean(iou_scores):.3f}",
        )
        ax1.set_xlabel("Frame Number", fontweight="bold", fontsize=11)
        ax1.set_ylabel("IoU Score", fontweight="bold", fontsize=11)
        ax1.set_title(
            f"IoU Scores (N={len(iou_scores)})", fontweight="bold", fontsize=12
        )
        ax1.grid(True, alpha=0.3)
        ax1.legend(fontsize=10)
        ax1.set_ylim([0, 1.1])

        # Add statistics text box
        stats_text = f"Mean: {np.mean(iou_scores):.3f}\nMedian: {np.median(iou_scores):.3f}\nStd: {np.std(iou_scores):.3f}"
        ax1.text(
            0.02,
            0.98,
            stats_text,
            transform=ax1.transAxes,
            fontsize=9,
            verticalalignment="top",
            bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5),
        )

    # DICE plot
    if dice_scores:
        ax2 = axes[1]
        ax2.plot(
            frames,
            dice_scores,
            "g-o",
            linewidth=2,
            markersize=6,
            alpha=0.7,
            label="DICE scores",
        )
        ax2.axhline(
            np.mean(dice_scores),
            color="red",
            linestyle="--",
            linewidth=2,
            label=f"Mean: {np.mean(dice_scores):.3f}",
        )
        ax2.set_xlabel("Frame Number", fontweight="bold", fontsize=11)
        ax2.set_ylabel("DICE Score", fontweight="bold", fontsize=11)
        ax2.set_title(
            f"DICE Scores (N={len(dice_scores)})", fontweight="bold", fontsize=12
        )
        ax2.grid(True, alpha=0.3)
        ax2.legend(fontsize=10)
        ax2.set_ylim([0, 1.1])

        # Add statistics text box
        stats_text = f"Mean: {np.mean(dice_scores):.3f}\nMedian: {np.median(dice_scores):.3f}\nStd: {np.std(dice_scores):.3f}"
        ax2.text(
            0.02,
            0.98,
            stats_text,
            transform=ax2.transAxes,
            fontsize=9,
            verticalalignment="top",
            bbox=dict(boxstyle="round", facecolor="lightgreen", alpha=0.5),
        )

    plt.tight_layout()

    # Save plot
    plot_filename = f"{safe_video_name}_evaluation.png"
    plot_path = os.path.join(output_dir, plot_filename)
    plt.savefig(plot_path, dpi=200, bbox_inches="tight")
    plt.close()

    print(f"    Saved per-video plot: {plot_filename}")


def evaluate_frame_batch(
    evaluator: TTIFrameEvaluator,
    frame_data_list: List[Tuple[str, int, List[Dict], str]],
    current_video_name: str = None,
    verbose: bool = True,
) -> Tuple[List[Dict], str]:
    """
    Evaluate a batch of frames.

    Args:
        frame_data_list: List of (video_path, frame_idx, ground_truth_objs_list, video_name)
        current_video_name: Name of current video being processed (for logging)
        verbose: Whether to print processing info

    Returns:
        Tuple of (List of evaluation results, current video name)
    """
    results = []

    for video_path, frame_idx, gt_objs, video_name in frame_data_list:
        # Print when switching to a new video
        if verbose and current_video_name != video_name:
            current_video_name = video_name
            print(f"  Processing video: {video_name}")

        # Extract frame
        frame = extract_frame(video_path, frame_idx)
        if frame is None:
            results.append(
                {
                    "video": video_name,
                    "frame": frame_idx,
                    "error": "Failed to extract frame",
                    "iou": None,
                    "dice": None,
                }
            )
            continue

        # Run inference once per frame
        predictions = evaluator.process_frame(frame, frame_idx)

        # If there are multiple ground truth boxes in this frame, match each with best prediction
        # and compute average IoU/DICE for the frame
        frame_ious = []
        frame_dices = []
        frame_matches = []

        for gt_obj in gt_objs:
            gt_bbox = gt_obj.get("boundingBox", {})
            gt_label = gt_obj.get("value", "")
            gt_is_tti = 1 if gt_label == "start_of_tti" else 0

            if not predictions:
                # No predictions for this ground truth
                frame_matches.append(
                    {
                        "video": video_name,
                        "frame": frame_idx,
                        "ground_truth_label": gt_label,
                        "ground_truth_tti": gt_is_tti,
                        "predicted_tti": None,
                        "iou": 0.0,
                        "dice": 0.0,
                        "matched": False,
                        "ground_truth_bbox": gt_bbox,
                    }
                )
                frame_ious.append(0.0)
                frame_dices.append(0.0)
                continue

            # Find best matching prediction by IoU for this ground truth box
            best_iou = 0.0
            best_dice = 0.0
            best_pred = None

            for pred in predictions:
                pred_bbox = pred["boundingBox"]
                iou = compute_iou(gt_bbox, pred_bbox)
                dice = compute_dice(gt_bbox, pred_bbox)

                if iou > best_iou:
                    best_iou = iou
                    best_dice = dice
                    best_pred = pred

            # Record match for this ground truth
            if best_pred:
                frame_matches.append(
                    {
                        "video": video_name,
                        "frame": frame_idx,
                        "ground_truth_label": gt_label,
                        "ground_truth_tti": gt_is_tti,
                        "predicted_tti": best_pred["tti_classification"],
                        "confidence": best_pred["confidence"],
                        "iou": best_iou,
                        "dice": best_dice,
                        "matched": True,
                        "ground_truth_bbox": gt_bbox,
                        "predicted_bbox": best_pred["boundingBox"],
                    }
                )
                frame_ious.append(best_iou)
                frame_dices.append(best_dice)
            else:
                frame_matches.append(
                    {
                        "video": video_name,
                        "frame": frame_idx,
                        "ground_truth_label": gt_label,
                        "ground_truth_tti": gt_is_tti,
                        "predicted_tti": None,
                        "iou": 0.0,
                        "dice": 0.0,
                        "matched": False,
                        "ground_truth_bbox": gt_bbox,
                    }
                )
                frame_ious.append(0.0)
                frame_dices.append(0.0)

        # Compute average IoU and DICE for this frame (across all ground truths)
        avg_iou = np.mean(frame_ious) if frame_ious else 0.0
        avg_dice = np.mean(frame_dices) if frame_dices else 0.0

        # Store all individual matches with the frame average
        for match in frame_matches:
            match["frame_avg_iou"] = avg_iou
            match["frame_avg_dice"] = avg_dice
            match["num_ground_truths_in_frame"] = len(gt_objs)
            results.append(match)

    return results, current_video_name


def generate_evaluation_plots(results: List[Dict], output_dir: str):
    """
    Generate IoU and DICE score plots per video and overall.

    Args:
        results: List of per-frame evaluation results
        output_dir: Directory to save plots
    """
    # Organize results by video
    video_results = defaultdict(list)
    for r in results:
        if r.get("iou") is not None:
            video_results[r["video"]].append(r)

    # Create figure with subplots
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle("TTI Detection Evaluation Results", fontsize=16, fontweight="bold")

    # 1. Per-video IoU scores (box plot)
    ax1 = axes[0, 0]
    video_names = list(video_results.keys())
    iou_per_video = [
        [r["iou"] for r in video_results[v] if r["iou"] is not None]
        for v in video_names
    ]

    if iou_per_video:
        bp1 = ax1.boxplot(
            iou_per_video,
            labels=[v[:20] + "..." if len(v) > 20 else v for v in video_names],
            patch_artist=True,
        )
        for patch in bp1["boxes"]:
            patch.set_facecolor("lightblue")
        ax1.set_xlabel("Video", fontweight="bold")
        ax1.set_ylabel("IoU Score", fontweight="bold")
        ax1.set_title("IoU Scores per Video", fontweight="bold")
        ax1.grid(True, alpha=0.3)
        ax1.tick_params(axis="x", rotation=45)

        # Add mean line
        means = [np.mean(scores) for scores in iou_per_video]
        ax1.plot(range(1, len(means) + 1), means, "ro-", label="Mean", markersize=5)
        ax1.legend()

    # 2. Per-video DICE scores (box plot)
    ax2 = axes[0, 1]
    dice_per_video = [
        [r["dice"] for r in video_results[v] if r["dice"] is not None]
        for v in video_names
    ]

    if dice_per_video:
        bp2 = ax2.boxplot(
            dice_per_video,
            labels=[v[:20] + "..." if len(v) > 20 else v for v in video_names],
            patch_artist=True,
        )
        for patch in bp2["boxes"]:
            patch.set_facecolor("lightgreen")
        ax2.set_xlabel("Video", fontweight="bold")
        ax2.set_ylabel("DICE Score", fontweight="bold")
        ax2.set_title("DICE Scores per Video", fontweight="bold")
        ax2.grid(True, alpha=0.3)
        ax2.tick_params(axis="x", rotation=45)

        # Add mean line
        means = [np.mean(scores) for scores in dice_per_video]
        ax2.plot(range(1, len(means) + 1), means, "ro-", label="Mean", markersize=5)
        ax2.legend()

    # 3. Overall IoU distribution (histogram)
    ax3 = axes[1, 0]
    all_iou = [r["iou"] for r in results if r["iou"] is not None and r["iou"] > 0]

    if all_iou:
        ax3.hist(all_iou, bins=30, color="skyblue", edgecolor="black", alpha=0.7)
        ax3.axvline(
            np.mean(all_iou),
            color="red",
            linestyle="--",
            linewidth=2,
            label=f"Mean: {np.mean(all_iou):.3f}",
        )
        ax3.axvline(
            np.median(all_iou),
            color="green",
            linestyle="--",
            linewidth=2,
            label=f"Median: {np.median(all_iou):.3f}",
        )
        ax3.set_xlabel("IoU Score", fontweight="bold")
        ax3.set_ylabel("Frequency", fontweight="bold")
        ax3.set_title(f"Overall IoU Distribution (n={len(all_iou)})", fontweight="bold")
        ax3.grid(True, alpha=0.3)
        ax3.legend()

    # 4. Overall DICE distribution (histogram)
    ax4 = axes[1, 1]
    all_dice = [r["dice"] for r in results if r["dice"] is not None and r["dice"] > 0]

    if all_dice:
        ax4.hist(all_dice, bins=30, color="lightcoral", edgecolor="black", alpha=0.7)
        ax4.axvline(
            np.mean(all_dice),
            color="red",
            linestyle="--",
            linewidth=2,
            label=f"Mean: {np.mean(all_dice):.3f}",
        )
        ax4.axvline(
            np.median(all_dice),
            color="green",
            linestyle="--",
            linewidth=2,
            label=f"Median: {np.median(all_dice):.3f}",
        )
        ax4.set_xlabel("DICE Score", fontweight="bold")
        ax4.set_ylabel("Frequency", fontweight="bold")
        ax4.set_title(
            f"Overall DICE Distribution (n={len(all_dice)})", fontweight="bold"
        )
        ax4.grid(True, alpha=0.3)
        ax4.legend()

    plt.tight_layout()

    # Save plot
    plot_path = os.path.join(output_dir, "evaluation_plots.png")
    plt.savefig(plot_path, dpi=300, bbox_inches="tight")
    print(f"\nPlots saved to: {plot_path}")
    plt.close()

    # Create per-video summary bar chart
    fig2, axes2 = plt.subplots(1, 2, figsize=(16, 6))
    fig2.suptitle("Per-Video Mean Scores", fontsize=16, fontweight="bold")

    # Mean IoU per video
    ax1 = axes2[0]
    mean_iou_per_video = [np.mean(scores) for scores in iou_per_video]
    colors = plt.cm.viridis(np.linspace(0, 1, len(video_names)))
    bars1 = ax1.bar(
        range(len(video_names)),
        mean_iou_per_video,
        color=colors,
        alpha=0.8,
        edgecolor="black",
    )
    ax1.set_xticks(range(len(video_names)))
    ax1.set_xticklabels(
        [v[:15] + "..." if len(v) > 15 else v for v in video_names],
        rotation=45,
        ha="right",
    )
    ax1.set_ylabel("Mean IoU Score", fontweight="bold")
    ax1.set_title("Mean IoU Score per Video", fontweight="bold")
    ax1.grid(True, alpha=0.3, axis="y")
    ax1.axhline(
        np.mean(mean_iou_per_video),
        color="red",
        linestyle="--",
        linewidth=2,
        label=f"Overall Mean: {np.mean(mean_iou_per_video):.3f}",
    )
    ax1.legend()

    # Add value labels on bars
    for i, (bar, val) in enumerate(zip(bars1, mean_iou_per_video)):
        height = bar.get_height()
        ax1.text(
            bar.get_x() + bar.get_width() / 2.0,
            height,
            f"{val:.3f}",
            ha="center",
            va="bottom",
            fontsize=8,
        )

    # Mean DICE per video
    ax2 = axes2[1]
    mean_dice_per_video = [np.mean(scores) for scores in dice_per_video]
    bars2 = ax2.bar(
        range(len(video_names)),
        mean_dice_per_video,
        color=colors,
        alpha=0.8,
        edgecolor="black",
    )
    ax2.set_xticks(range(len(video_names)))
    ax2.set_xticklabels(
        [v[:15] + "..." if len(v) > 15 else v for v in video_names],
        rotation=45,
        ha="right",
    )
    ax2.set_ylabel("Mean DICE Score", fontweight="bold")
    ax2.set_title("Mean DICE Score per Video", fontweight="bold")
    ax2.grid(True, alpha=0.3, axis="y")
    ax2.axhline(
        np.mean(mean_dice_per_video),
        color="red",
        linestyle="--",
        linewidth=2,
        label=f"Overall Mean: {np.mean(mean_dice_per_video):.3f}",
    )
    ax2.legend()

    # Add value labels on bars
    for i, (bar, val) in enumerate(zip(bars2, mean_dice_per_video)):
        height = bar.get_height()
        ax2.text(
            bar.get_x() + bar.get_width() / 2.0,
            height,
            f"{val:.3f}",
            ha="center",
            va="bottom",
            fontsize=8,
        )

    plt.tight_layout()

    # Save per-video summary plot
    summary_plot_path = os.path.join(output_dir, "per_video_summary.png")
    plt.savefig(summary_plot_path, dpi=300, bbox_inches="tight")
    print(f"Per-video summary saved to: {summary_plot_path}")
    plt.close()


def main():
    parser = argparse.ArgumentParser(description="TTI Frame-Level Evaluation Script")
    parser.add_argument(
        "--annotations_dir",
        default=DEFAULT_GROUND_TRUTH_PATH,
        help="Directory containing annotation JSON files",
    )
    parser.add_argument(
        "--videos_dir",
        default=DEFAULT_VIDEO_PATH,
        help="Directory containing video files",
    )
    parser.add_argument(
        "--output",
        default=os.path.join(
            os.path.dirname(os.path.abspath(__file__)), "results.json"
        ),
        help="Path to save evaluation results JSON",
    )
    parser.add_argument(
        "--vit_model", default=DEFAULT_VIT_MODEL_PATH, help="Path to ViT model"
    )
    parser.add_argument("--yolo_model", default=None, help="Path to YOLO model")
    parser.add_argument("--device", default=None, help="Device to use (cuda/cpu/mps)")
    parser.add_argument(
        "--batch_size",
        type=int,
        default=DEFAULT_BATCH_SIZE,
        help="Batch size for processing",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=DEFAULT_WORKERS,
        help="Number of parallel workers",
    )

    args = parser.parse_args()

    # Validate paths
    if not os.path.isdir(args.annotations_dir):
        print(f"Error: Annotations directory not found: {args.annotations_dir}")
        return

    if not os.path.isdir(args.videos_dir):
        print(f"Error: Videos directory not found: {args.videos_dir}")
        return

    if not os.path.exists(args.vit_model):
        print(f"Error: ViT model not found: {args.vit_model}")
        return

    # Create output directory
    output_dir = os.path.dirname(args.output)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    print("=" * 80)
    print("TTI Frame-Level Evaluation")
    print("=" * 80)
    print("\nConfiguration:")
    print(f"  Annotations: {args.annotations_dir}")
    print(f"  Videos: {args.videos_dir}")
    print(f"  ViT Model: {args.vit_model}")
    print(f"  Output: {args.output}")
    print(f"  Batch Size: {args.batch_size}")

    # Load ground truth annotations
    print(f"\nLoading annotations from: {args.annotations_dir}")
    annotations_map = load_ground_truth_annotations(args.annotations_dir)
    print(f"Loaded {len(annotations_map)} annotation files")

    # Extract TTI frames and group by (video, frame) to handle multiple ground truths per frame
    print("\nExtracting TTI frames from annotations...")
    all_tti_frames = []
    frame_groups = defaultdict(
        list
    )  # Key: (video_path, frame_idx, video_name), Value: list of gt_objs

    for video_name, annotation_data in annotations_map.items():
        tti_frames = extract_tti_frames_from_annotations(annotation_data)

        # Find corresponding video file
        video_path = os.path.join(args.videos_dir, video_name)
        if not os.path.exists(video_path):
            # Try to find by matching filename
            possible_videos = glob.glob(
                os.path.join(args.videos_dir, f"*{Path(video_name).stem}*")
            )
            if possible_videos:
                video_path = possible_videos[0]
            else:
                print(f"Warning: Video not found for {video_name}, skipping")
                continue

        # Group ground truth objects by frame
        for frame_idx, label_type, gt_obj in tti_frames:
            key = (video_path, frame_idx, video_name)
            frame_groups[key].append(gt_obj)

    # Convert grouped frames to list format for batch processing
    for (video_path, frame_idx, video_name), gt_objs in frame_groups.items():
        all_tti_frames.append((video_path, frame_idx, gt_objs, video_name))

    print(f"Found {len(all_tti_frames)} TTI frames to evaluate")

    if len(all_tti_frames) == 0:
        print("No TTI frames found. Exiting.")
        return

    # Initialize evaluator
    print("\nInitializing TTI evaluator...")
    evaluator = TTIFrameEvaluator(
        vit_model_path=args.vit_model,
        yolo_model_path=args.yolo_model,
        device=args.device,
        focus_ratio=DEFAULT_FOCUS_RATIO,
        use_half_precision=True,
    )

    print(f"\n{'=' * 80}")
    print(f"Device being used: {evaluator.device.upper()}")
    print(f"{'=' * 80}")
    print(f"Batch size: {args.batch_size}")
    print(f"Half precision (ViT): {evaluator.use_half_precision_vit}")
    print(f"Half precision (YOLO): {evaluator.use_half_precision_yolo}")

    # Process frames in batches and track per-video results
    print(f"\n{'=' * 80}")
    print(f"Processing {len(all_tti_frames)} TTI frames...")
    print(f"{'=' * 80}")
    all_results = []
    current_video = None
    video_results_buffer = defaultdict(list)  # Buffer results per video

    for i in tqdm(
        range(0, len(all_tti_frames), args.batch_size), desc="Evaluating frames"
    ):
        batch = all_tti_frames[i : i + args.batch_size]
        prev_video = current_video
        batch_results, current_video = evaluate_frame_batch(
            evaluator, batch, current_video, verbose=True
        )

        # Add results to buffer
        for result in batch_results:
            video_name = result.get("video", "unknown")
            video_results_buffer[video_name].append(result)

        all_results.extend(batch_results)

        # When we finish a video (detected by video name change), save its plot
        if (
            prev_video is not None
            and current_video != prev_video
            and prev_video in video_results_buffer
        ):
            output_plot_dir = os.path.dirname(args.output) or "."
            save_per_video_plot(
                prev_video, video_results_buffer[prev_video], output_plot_dir
            )

    # Save plot for the last video
    if current_video is not None and current_video in video_results_buffer:
        output_plot_dir = os.path.dirname(args.output) or "."
        save_per_video_plot(
            current_video, video_results_buffer[current_video], output_plot_dir
        )

    # Compute statistics
    print("\n" + "=" * 80)
    print("EVALUATION RESULTS")
    print("=" * 80)

    # Deduplicate results by frame to get frame-level averages for statistics
    # (since multiple ground truths per frame will have the same frame_avg values)
    frame_level_results = {}
    for r in all_results:
        frame_key = (r.get("video"), r.get("frame"))
        if frame_key not in frame_level_results:
            # Use frame averages for IoU/DICE statistics
            frame_level_results[frame_key] = {
                "video": r.get("video"),
                "frame": r.get("frame"),
                "iou": r.get(
                    "frame_avg_iou", r.get("iou")
                ),  # Fallback to individual if avg not present
                "dice": r.get("frame_avg_dice", r.get("dice")),
                "ground_truth_tti": r.get("ground_truth_tti"),
                "predicted_tti": r.get("predicted_tti"),
            }

    # Compute per-video statistics using frame-level data
    video_stats = defaultdict(
        lambda: {"iou": [], "dice": [], "tp": 0, "fp": 0, "tn": 0, "fn": 0}
    )
    for r in frame_level_results.values():
        video_name = r.get("video", "unknown")
        if r.get("iou") is not None:
            video_stats[video_name]["iou"].append(r["iou"])
        if r.get("dice") is not None:
            video_stats[video_name]["dice"].append(r["dice"])

        # Confusion matrix per video
        gt_tti = r.get("ground_truth_tti")
        pred_tti = r.get("predicted_tti")
        if gt_tti == 1 and pred_tti == 1:
            video_stats[video_name]["tp"] += 1
        elif gt_tti == 0 and pred_tti == 1:
            video_stats[video_name]["fp"] += 1
        elif gt_tti == 0 and pred_tti == 0:
            video_stats[video_name]["tn"] += 1
        elif gt_tti == 1 and pred_tti == 0:
            video_stats[video_name]["fn"] += 1

    # Use frame-level results for overall statistics
    iou_scores = [
        r["iou"] for r in frame_level_results.values() if r["iou"] is not None
    ]
    dice_scores = [
        r["dice"] for r in frame_level_results.values() if r["dice"] is not None
    ]

    # TTI classification accuracy (frame level)
    correct_tti = sum(
        1
        for r in frame_level_results.values()
        if r.get("predicted_tti") == r.get("ground_truth_tti")
    )
    total_with_pred = sum(
        1 for r in frame_level_results.values() if r.get("predicted_tti") is not None
    )

    # Confusion matrix (frame level)
    tp = sum(
        1
        for r in frame_level_results.values()
        if r.get("ground_truth_tti") == 1 and r.get("predicted_tti") == 1
    )
    fp = sum(
        1
        for r in frame_level_results.values()
        if r.get("ground_truth_tti") == 0 and r.get("predicted_tti") == 1
    )
    tn = sum(
        1
        for r in frame_level_results.values()
        if r.get("ground_truth_tti") == 0 and r.get("predicted_tti") == 0
    )
    fn = sum(
        1
        for r in frame_level_results.values()
        if r.get("ground_truth_tti") == 1 and r.get("predicted_tti") == 0
    )

    print(f"\nTotal ground truth annotations: {len(all_results)}")
    print(f"Total unique frames evaluated: {len(frame_level_results)}")
    print(f"Frames with predictions: {total_with_pred}")
    print(f"Frames without predictions: {len(frame_level_results) - total_with_pred}")

    # Print per-video statistics
    print(f"\n{'=' * 80}")
    print("PER-VIDEO STATISTICS")
    print(f"{'=' * 80}")
    for video_name in sorted(video_stats.keys()):
        stats = video_stats[video_name]
        print(f"\n{video_name}:")
        if stats["iou"]:
            print(
                f"  IoU  - Mean: {np.mean(stats['iou']):.4f}, Median: {np.median(stats['iou']):.4f}, "
                f"Std: {np.std(stats['iou']):.4f}, N: {len(stats['iou'])}"
            )
        if stats["dice"]:
            print(
                f"  DICE - Mean: {np.mean(stats['dice']):.4f}, Median: {np.median(stats['dice']):.4f}, "
                f"Std: {np.std(stats['dice']):.4f}, N: {len(stats['dice'])}"
            )
        print(
            f"  Confusion Matrix - TP: {stats['tp']}, FP: {stats['fp']}, TN: {stats['tn']}, FN: {stats['fn']}"
        )

    print(f"\n{'=' * 80}")
    print("OVERALL STATISTICS")
    print(f"{'=' * 80}")

    print("\n--- IoU Scores ---")
    if iou_scores:
        print(f"Mean IoU: {np.mean(iou_scores):.4f}")
        print(f"Std IoU: {np.std(iou_scores):.4f}")
        print(f"Min IoU: {np.min(iou_scores):.4f}")
        print(f"Max IoU: {np.max(iou_scores):.4f}")
        print(f"Median IoU: {np.median(iou_scores):.4f}")

    print("\n--- DICE Scores ---")
    if dice_scores:
        print(f"Mean DICE: {np.mean(dice_scores):.4f}")
        print(f"Std DICE: {np.std(dice_scores):.4f}")
        print(f"Min DICE: {np.min(dice_scores):.4f}")
        print(f"Max DICE: {np.max(dice_scores):.4f}")
        print(f"Median DICE: {np.median(dice_scores):.4f}")

    print("\n--- TTI Classification ---")
    if total_with_pred > 0:
        print(
            f"Accuracy: {correct_tti / total_with_pred * 100:.2f}% ({correct_tti}/{total_with_pred})"
        )

    print("\n--- Confusion Matrix ---")
    print(f"True Positives (TP): {tp}")
    print(f"False Positives (FP): {fp}")
    print(f"True Negatives (TN): {tn}")
    print(f"False Negatives (FN): {fn}")

    if tp + fp > 0:
        precision = tp / (tp + fp)
        print(f"\nPrecision: {precision:.4f}")
    else:
        precision = None

    if tp + fn > 0:
        recall = tp / (tp + fn)
        print(f"Recall: {recall:.4f}")
    else:
        recall = None

    if precision is not None and recall is not None and (precision + recall) > 0:
        f1 = 2 * precision * recall / (precision + recall)
        print(f"F1 Score: {f1:.4f}")
    else:
        f1 = None

    # Save results with per-video statistics
    per_video_summary = {}
    for video_name, stats in video_stats.items():
        per_video_summary[video_name] = {
            "iou_mean": float(np.mean(stats["iou"])) if stats["iou"] else None,
            "iou_median": float(np.median(stats["iou"])) if stats["iou"] else None,
            "iou_std": float(np.std(stats["iou"])) if stats["iou"] else None,
            "iou_min": float(np.min(stats["iou"])) if stats["iou"] else None,
            "iou_max": float(np.max(stats["iou"])) if stats["iou"] else None,
            "dice_mean": float(np.mean(stats["dice"])) if stats["dice"] else None,
            "dice_median": float(np.median(stats["dice"])) if stats["dice"] else None,
            "dice_std": float(np.std(stats["dice"])) if stats["dice"] else None,
            "dice_min": float(np.min(stats["dice"])) if stats["dice"] else None,
            "dice_max": float(np.max(stats["dice"])) if stats["dice"] else None,
            "num_frames": len(stats["iou"]),
            "confusion_matrix": {
                "tp": stats["tp"],
                "fp": stats["fp"],
                "tn": stats["tn"],
                "fn": stats["fn"],
            },
        }

    output_data = {
        "summary": {
            "total_ground_truth_annotations": len(all_results),
            "total_unique_frames": len(frame_level_results),
            "frames_with_predictions": total_with_pred,
            "iou_mean": float(np.mean(iou_scores)) if iou_scores else None,
            "iou_std": float(np.std(iou_scores)) if iou_scores else None,
            "iou_min": float(np.min(iou_scores)) if iou_scores else None,
            "iou_max": float(np.max(iou_scores)) if iou_scores else None,
            "iou_median": float(np.median(iou_scores)) if iou_scores else None,
            "dice_mean": float(np.mean(dice_scores)) if dice_scores else None,
            "dice_std": float(np.std(dice_scores)) if dice_scores else None,
            "dice_min": float(np.min(dice_scores)) if dice_scores else None,
            "dice_max": float(np.max(dice_scores)) if dice_scores else None,
            "dice_median": float(np.median(dice_scores)) if dice_scores else None,
            "tti_accuracy": correct_tti / total_with_pred
            if total_with_pred > 0
            else None,
            "confusion_matrix": {"tp": tp, "fp": fp, "tn": tn, "fn": fn},
            "precision": precision,
            "recall": recall,
            "f1_score": f1,
        },
        "per_video_summary": per_video_summary,
        "per_frame_results": all_results,
    }

    print(f"\n{'=' * 80}")
    print(f"Saving results to: {args.output}")
    with open(args.output, "w") as f:
        json.dump(output_data, f, indent=2)
    print("Results saved successfully!")

    # Generate plots
    print(f"\n{'=' * 80}")
    print("Generating evaluation plots...")
    print(f"{'=' * 80}")
    output_dir = os.path.dirname(args.output) or "."
    generate_evaluation_plots(all_results, output_dir)

    print(f"\n{'=' * 80}")
    print("EVALUATION COMPLETE!")
    print(f"{'=' * 80}")
    print("\nOutput files:")
    print(f"  - Results JSON: {args.output}")
    print(f"  - Plots: {os.path.join(output_dir, 'evaluation_plots.png')}")
    print(f"  - Per-video summary: {os.path.join(output_dir, 'per_video_summary.png')}")


if __name__ == "__main__":
    main()
