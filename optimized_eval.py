#!/usr/bin/env python3
"""
Optimized ViT TTI Classifier Evaluation Script with Complete Visualization Control

Usage:
  # For video
  python optimized_eval.py --video /path/to/video.mp4 --output /path/to/results.json [--output_video /path/to/annotated.mp4]

  # For single image
  python optimized_eval.py --image /path/to/image.jpg --output /path/to/results.json [--output_image /path/to/annotated.jpg]

  # Common options
  [--show_heatmap] [--show_roi_box] [--vit_model PATH] [--yolo_model PATH] [--device cuda|cpu|mps]

Arguments:
- --video: Path to the input video file to analyze.
- --image: Path to the input image file to analyze.
- --output: Path to the output JSON file where detections/metrics are saved. Required.
- --output_video: Optional path to save an annotated MP4 of the run (for video input).
- --output_image: Optional path to save an annotated image of the run (for image input).
- --show_heatmap: If set, overlays per-class heat maps (blue=tools, green=tissue) on frames.
- --show_roi_box: If set, draws ROI bounding boxes used for classification.
- --vit_model: Filesystem path to the ViT classifier weights. Default: value of DEFAULT_VIT_MODEL_PATH.
- --yolo_model: Filesystem path to the YOLO segmentation/detection weights. Default: value of DEFAULT_YOLO_MODEL_PATH.
- --start_frame: Index of the first frame to process (0-based, for video). Default: 0.
- --end_frame: Index of the last frame to process (inclusive, for video). Default: None (process to end).
- --frame_step: Process every Nth frame (for video); 1 = real-time/full rate. Default: DEFAULT_FRAME_STEP.
- --device: Compute device to use: cuda, cpu, or mps. Default: auto-detect when None.
- --focus_ratio: ROI focus ratio controlling crop size around detections. Default: DEFAULT_FOCUS_RATIO.
- --depth_model: Optional local path to a depth model; if None, uses a default model id. Default: None.
- --batch_size: Number of frames to process per batch across stages. Default: DEFAULT_BATCH_SIZE.
- --disable_half_precision: If set, disables FP16 inference (forces full precision). Default: off (FP16 enabled when supported).
"""

import argparse
import json
import os
import uuid
import warnings
from datetime import datetime
from typing import Dict, List, Tuple

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from tqdm import tqdm
from transformers import pipeline

warnings.filterwarnings("ignore")

# Import local modules
from ultralytics import YOLO

"""
Global configuration constants
All configurable values, paths, sizes, colors, and defaults are centralized here.
"""

# Model paths and identifiers
DEFAULT_VIT_MODEL_PATH = "/cluster/projects/madanigroup/lorenz/tti/vit.pt"
DEFAULT_YOLO_MODEL_PATH = "/cluster/projects/madanigroup/lorenz/tti/yolo.pt"
DEFAULT_DEPTH_MODEL_ID = "Intel/dpt-large"

# Processing defaults
DEFAULT_BATCH_SIZE = 8
DEFAULT_FRAME_STEP = 1  # Real-time by default
DEFAULT_FOCUS_RATIO = 1.0
DEFAULT_USE_HALF_PRECISION = True

# Caches and thresholds
DEPTH_CACHE_MAX_SIZE = 50
ROI_MIN_SIZE = 64

# ViT architecture/config defaults
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

# Visualization settings
HEATMAP_ALPHA = 0.4
HEATMAP_TOOL_COLOR = (0.0, 0.0, 1.0)  # blue
HEATMAP_TISSUE_COLOR = (0.0, 1.0, 0.0)  # green

# TTI label settings
TTI_CLASS_POSITIVE_NAME = "Start of TTI"
TTI_CLASS_POSITIVE_VALUE = "start_of_tti"
TTI_CLASS_POSITIVE_COLOR = "#D33115"
TTI_CLASS_NEGATIVE_NAME = "Start of No Interaction"
TTI_CLASS_NEGATIVE_VALUE = "start_of_no_interaction"
TTI_CLASS_NEGATIVE_COLOR = "#1eff00"

# Class mappings and names
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


def show_mask_overlay_from_binary_mask(
    image_bgr, binary_mask, alpha=0.5, mask_color=(1.0, 0.0, 0.0)
):
    """Create a colored overlay from a binary mask"""
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    colored_mask = np.zeros_like(image_rgb)
    colored_mask[..., 0] = mask_color[0]
    colored_mask[..., 1] = mask_color[1]
    colored_mask[..., 2] = mask_color[2]

    overlay = image_rgb.copy()
    indices = binary_mask.astype(bool)
    overlay[indices] = (1 - alpha) * image_rgb[indices] + alpha * colored_mask[indices]

    return cv2.cvtColor((overlay * 255).astype(np.uint8), cv2.COLOR_RGB2BGR)


class OptimizedTTIVideoEvaluator:
    """Optimized Video evaluator for TTI classification using batch processing with full visualization"""

    def __init__(
        self,
        vit_model_path,
        yolo_model_path=None,
        device=None,
        focus_ratio=1.0,
        depth_model_path=None,
        batch_size=8,
        use_half_precision=True,
    ):
        """Initialize the optimized TTI evaluator"""
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
        self.batch_size = batch_size

        # Bind class mappings and names to instance from module-level constants
        self.TOOL_CLASSES = TOOL_CLASSES
        self.TTI_CLASSES = TTI_CLASSES
        self.TOOL_NAMES = TOOL_NAMES
        self.TTI_NAMES = TTI_NAMES

        # Disable half precision for YOLO on CUDA due to known issues
        if self.device == "cuda":
            self.use_half_precision_vit = use_half_precision
            self.use_half_precision_yolo = False
            print(
                "CUDA detected: Disabling half precision for YOLO to avoid dtype conflicts"
            )
        else:
            self.use_half_precision_vit = False
            self.use_half_precision_yolo = False

        print(f"Using device: {self.device}")
        print(f"Batch size: {batch_size}")
        print(f"ViT half precision: {self.use_half_precision_vit}")
        print(f"YOLO half precision: {self.use_half_precision_yolo}")

        # Enable optimizations
        if self.device == "cuda":
            torch.backends.cudnn.benchmark = True
            torch.backends.cudnn.deterministic = False

        # Load models
        self.tti_model = self._load_tti_model(vit_model_path)
        self.yolo_model = self._load_yolo_model(yolo_model_path)
        self._load_depth_model(depth_model_path)

        # Cache for depth maps to avoid recomputation
        self.depth_cache = {}

        print("All models loaded successfully!")

    def _load_tti_model(self, model_path):
        """Load the trained ViT TTI classifier with optimizations"""
        print(f"Loading ViT TTI classifier from: {model_path}")

        model = self._create_offline_vit_model(num_hoi_classes=2)
        checkpoint = torch.load(model_path, map_location=self.device)
        model.load_state_dict(checkpoint)
        model.to(self.device)

        if self.use_half_precision_vit:
            model.half()
            print("ViT model converted to half precision")

        model.eval()
        return model

    def _create_offline_vit_model(self, num_hoi_classes):
        """Create optimized ROIClassifierViT"""
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

        return OptimizedROIClassifierViT(num_hoi_classes)

    def _load_yolo_model(self, model_path):
        """Load YOLO model with CUDA-safe initialization"""
        print(f"Loading YOLO model from: {model_path}")
        model = YOLO(model_path)

        # Force YOLO to use float32 to avoid half precision issues
        if self.device == "cuda":
            try:
                model.model.float()
                print("YOLO model forced to float32 for CUDA stability")
            except Exception as e:
                print(f"Warning: Could not force YOLO to float32: {e}")

        return model

    def _load_depth_model(self, depth_model_path):
        """Load depth estimation model"""
        print("Loading depth estimation model...")
        try:
            if depth_model_path and os.path.exists(depth_model_path):
                device_id = 0 if self.device in ["cuda", "mps"] else -1
                self.depth_model = pipeline(
                    task="depth-estimation",
                    model=depth_model_path,
                    device=device_id,
                )
            else:
                device_id = 0 if self.device in ["cuda", "mps"] else -1
                self.depth_model = pipeline(
                    task="depth-estimation",
                    model=DEFAULT_DEPTH_MODEL_ID,
                    device=device_id,
                )
            self.use_real_depth = True
            print("Depth model loaded successfully!")
        except Exception as e:
            print(f"Using fallback depth estimation: {e}")
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

    def process_frames_batch(
        self,
        frames: List[np.ndarray],
        frame_indices: List[int],
        draw_annotations=False,
        show_heatmap=True,
        show_roi_box=False,
    ) -> Tuple[List[Dict], List[np.ndarray]]:
        """Process a batch of frames efficiently with corrected visualization logic"""
        batch_results = []
        annotated_frames = []

        # Step 1: YOLO inference with proper error handling
        try:
            yolo_results_batch = self.yolo_model(frames, verbose=False)
        except Exception as e:
            print(f"Batch YOLO inference failed: {e}")
            print("Falling back to single frame processing...")
            yolo_results_batch = []
            for i, frame in enumerate(frames):
                try:
                    result = self.yolo_model(frame, verbose=False)
                    yolo_results_batch.append(
                        result[0] if isinstance(result, list) else result
                    )
                except Exception as single_e:
                    print(
                        f"Single frame YOLO failed for frame {frame_indices[i]}: {single_e}"
                    )
                    yolo_results_batch.append(None)

        # Step 2: Process each frame in the batch
        for i, (frame, frame_idx) in enumerate(zip(frames, frame_indices)):
            if i >= len(yolo_results_batch) or yolo_results_batch[i] is None:
                frame_result = {
                    "frame": frame_idx,
                    "objects": [],
                    "classifications": [],
                }
                annotated_frame = (
                    frame.copy() if (draw_annotations or show_heatmap) else None
                )
                batch_results.append(frame_result)
                annotated_frames.append(annotated_frame)
                continue

            # Get depth map (with caching)
            depth_cache_key = f"{frame_idx}"
            if depth_cache_key in self.depth_cache:
                depth_map = self.depth_cache[depth_cache_key]
            else:
                pil_frame = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                if self.use_real_depth and self.depth_model is not None:
                    try:
                        depth_result = self.depth_model(pil_frame)
                        depth_map = np.array(depth_result["depth"])
                    except Exception as e:
                        print(
                            f"Depth estimation failed for frame {frame_idx}, using fallback: {e}"
                        )
                        depth_map = self._fallback_depth_estimation(pil_frame)
                else:
                    depth_map = self._fallback_depth_estimation(pil_frame)

                self.depth_cache[depth_cache_key] = depth_map
                if len(self.depth_cache) > DEPTH_CACHE_MAX_SIZE:
                    oldest_key = min(self.depth_cache.keys())
                    del self.depth_cache[oldest_key]

            # Parse YOLO output
            detections = self.parse_yolo_output([yolo_results_batch[i]])
            pairs = self.find_tool_tissue_pairs_fast(detections)

            frame_result = {"frame": frame_idx, "objects": [], "classifications": []}
            annotated_frame = (
                frame.copy() if (draw_annotations or show_heatmap) else None
            )

            # Collect ROIs for batch ViT inference FIRST
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
                    pair_info.append((pair, bbox, frame_idx))

            # Step 3: Batch ViT inference to get TTI classifications
            tti_results_map = {}  # Map pair indices to TTI results

            if roi_batch:
                try:
                    roi_batch_tensor = torch.stack(roi_batch).to(self.device)

                    with torch.no_grad():
                        logits_batch = self.tti_model(roi_batch_tensor)
                        probabilities_batch = F.softmax(logits_batch, dim=1)
                        tti_classes = torch.argmax(logits_batch, dim=1).cpu().numpy()
                        tti_confidences = (
                            probabilities_batch.max(dim=1)[0].cpu().numpy()
                        )

                    # Process results and store TTI classifications
                    for j, (pair, bbox, frame_idx) in enumerate(pair_info):
                        tti_class = tti_classes[j]
                        tti_confidence = tti_confidences[j]

                        tti_result = self._create_tti_result(
                            pair,
                            bbox,
                            frame_idx,
                            tti_class,
                            tti_confidence,
                            frame.shape,
                        )

                        if tti_result:
                            frame_result["objects"].append(tti_result)
                            # Store TTI classification for this pair
                            tti_results_map[j] = {
                                "tti_class": tti_class,
                                "tti_confidence": tti_confidence,
                                "pair": pair,
                            }

                except Exception as e:
                    print(f"ViT inference failed for frame {frame_idx}: {e}")

            # Deduplicate results
            if frame_result["objects"]:
                frame_result["objects"] = self._deduplicate_final_results_fast(
                    frame_result["objects"]
                )

            # FIXED HEAT MAP LOGIC: Only show green for ACTUAL TTI detections
            if show_heatmap and annotated_frame is not None and pairs:
                H_full, W_full = frame.shape[:2]
                combined_tool_mask = np.zeros((H_full, W_full), dtype=np.uint8)
                combined_tissue_mask = np.zeros((H_full, W_full), dtype=np.uint8)

                # Only show masks for pairs that have TTI detected (tti_class == 1)
                for j, (pair, bbox, frame_idx_inner) in enumerate(pair_info):
                    # Get TTI classification for this specific pair
                    tti_info = tti_results_map.get(j)
                    if tti_info and tti_info["tti_class"] == 1:  # Only for TTI detected
                        tool_mask = pair["tool"]["mask"]
                        tissue_mask = pair["tissue"]["mask"]

                        tool_mask_resized = cv2.resize(
                            tool_mask.astype(np.uint8),
                            (W_full, H_full),
                            interpolation=cv2.INTER_NEAREST,
                        )
                        tissue_mask_resized = cv2.resize(
                            tissue_mask.astype(np.uint8),
                            (W_full, H_full),
                            interpolation=cv2.INTER_NEAREST,
                        )

                        combined_tool_mask = np.maximum(
                            combined_tool_mask, tool_mask_resized
                        )
                        combined_tissue_mask = np.maximum(
                            combined_tissue_mask, tissue_mask_resized
                        )

                # Apply heat map overlays only for TTI detected cases
                if np.any(combined_tool_mask):
                    overlay_image_tool = show_mask_overlay_from_binary_mask(
                        annotated_frame,
                        combined_tool_mask,
                        alpha=HEATMAP_ALPHA,
                        mask_color=HEATMAP_TOOL_COLOR,
                    )
                    annotated_frame = overlay_image_tool

                if np.any(combined_tissue_mask):
                    overlay_image_tissue = show_mask_overlay_from_binary_mask(
                        annotated_frame,
                        combined_tissue_mask,
                        alpha=HEATMAP_ALPHA,
                        mask_color=HEATMAP_TISSUE_COLOR,
                    )
                    annotated_frame = overlay_image_tissue

            # Add detailed TTI labels
            if draw_annotations and annotated_frame is not None:
                for idx, tti_result in enumerate(frame_result["objects"]):
                    self._draw_detailed_tti_label(
                        annotated_frame,
                        tti_result,
                        label_index=idx,
                        show_roi_box=show_roi_box,
                    )

            batch_results.append(frame_result)
            annotated_frames.append(annotated_frame)

        return batch_results, annotated_frames

    def _draw_detailed_tti_label(
        self, frame, tti_result, label_index=0, show_roi_box=False
    ):
        """
        Draw detailed TTI label with tool and tissue information

        Args:
            frame (np.ndarray): Frame to draw on (modified in place)
            tti_result (dict): TTI detection result
            label_index (int): Index of this label (for positioning multiple labels)
            show_roi_box (bool): Whether to draw the ROI bounding box
        """
        # Get tool and tissue names
        tool_name = tti_result["tool_info"]["name"]
        tissue_name = tti_result["tissue_info"]["name"]
        tti_confidence = tti_result["confidence"]

        # Create detailed label text based on TTI classification
        if tti_result["tti_classification"] == 1:
            main_label = "TTI DETECTED"
            detail_text = f"{tool_name} + {tissue_name}"
            confidence_text = f"Conf: {tti_confidence:.2f}"
            label_color = (0, 255, 0)  # Green background for TTI
            text_color = (0, 0, 0)  # Black text
            box_color = (0, 255, 0)  # Green box for TTI
        else:
            main_label = "NO TTI"
            detail_text = f"{tool_name} + {tissue_name}"
            confidence_text = f"Conf: {tti_confidence:.2f}"
            label_color = (0, 0, 255)  # Red background for no TTI
            text_color = (255, 255, 255)  # White text
            box_color = (0, 0, 255)  # Red box for no TTI

        # Get frame dimensions
        h, w = frame.shape[:2]

        # Get bounding box coordinates (normalized to pixel coordinates)
        bbox = tti_result["boundingBox"]
        bbox_x = int(bbox["x"] * w)
        bbox_y = int(bbox["y"] * h)
        bbox_w = int(bbox["w"] * w)
        bbox_h = int(bbox["h"] * h)

        # Draw the interaction bounding box ONLY if show_roi_box is True
        if show_roi_box:
            cv2.rectangle(
                frame,
                (bbox_x, bbox_y),
                (bbox_x + bbox_w, bbox_y + bbox_h),
                box_color,
                2,
            )

        # Calculate text sizes for background
        font = cv2.FONT_HERSHEY_SIMPLEX
        main_font_scale = 0.7
        detail_font_scale = 0.5
        thickness = 2

        (main_w, main_h), _ = cv2.getTextSize(
            main_label, font, main_font_scale, thickness
        )
        (detail_w, detail_h), _ = cv2.getTextSize(
            detail_text, font, detail_font_scale, thickness
        )
        (conf_w, conf_h), _ = cv2.getTextSize(
            confidence_text, font, detail_font_scale, thickness
        )

        # Calculate text background dimensions
        text_bg_w = max(main_w, detail_w, conf_w) + 12
        text_bg_h = main_h + detail_h + conf_h + 18

        # Position label above the bounding box, but ensure it's within frame bounds
        label_x = bbox_x + bbox_w // 2 - text_bg_w // 2  # Center horizontally on bbox
        label_y = bbox_y - text_bg_h - 10  # Above the bbox with some margin

        # Adjust if label would go outside frame
        if label_y < 0:
            label_y = bbox_y + bbox_h + 10  # Place below bbox instead
        if label_x < 0:
            label_x = 10
        elif label_x + text_bg_w > w:
            label_x = w - text_bg_w - 10
        if label_y + text_bg_h > h:
            label_y = h - text_bg_h - 10

        # Offset multiple labels to avoid overlap
        if label_index > 0:
            offset_x = (label_index * 20) % 100
            offset_y = (label_index * 15) % 50
            label_x = max(10, min(w - text_bg_w - 10, label_x + offset_x))
            label_y = max(10, min(h - text_bg_h - 10, label_y + offset_y))

        # Draw background rectangle for text with border
        bg_y1 = label_y
        bg_y2 = label_y + text_bg_h

        # Draw filled background
        cv2.rectangle(
            frame, (label_x, bg_y1), (label_x + text_bg_w, bg_y2), label_color, -1
        )
        # Draw border for better visibility
        cv2.rectangle(
            frame, (label_x, bg_y1), (label_x + text_bg_w, bg_y2), (0, 0, 0), 2
        )

        # Draw text lines
        line1_y = bg_y1 + main_h + 6
        line2_y = line1_y + detail_h + 6
        line3_y = line2_y + conf_h + 6

        # Main label (TTI DETECTED / NO TTI)
        cv2.putText(
            frame,
            main_label,
            (label_x + 6, line1_y),
            font,
            main_font_scale,
            text_color,
            thickness,
        )

        # Tool -> Tissue detail
        cv2.putText(
            frame,
            detail_text,
            (label_x + 6, line2_y),
            font,
            detail_font_scale,
            text_color,
            thickness,
        )

        # Confidence
        cv2.putText(
            frame,
            confidence_text,
            (label_x + 6, line3_y),
            font,
            detail_font_scale,
            text_color,
            thickness,
        )

    def find_tool_tissue_pairs_fast(self, detections):
        """Fast tool-tissue pair finding with simplified deduplication"""
        tools = [d for d in detections if d["type"] == "tool"]
        tissues = [d for d in detections if d["type"] == "tti"]

        pairs = []
        for tool in tools:
            for tissue in tissues:
                pairs.append({"tool": tool, "tissue": tissue})

        # Simple deduplication based on confidence
        if len(pairs) > 3:  # Only deduplicate if too many pairs
            pairs = sorted(
                pairs,
                key=lambda p: p["tool"]["confidence"] + p["tissue"]["confidence"],
                reverse=True,
            )[:3]  # Keep top 3 pairs

        return pairs

    def extract_union_roi_fast(
        self, image, tool_mask, tissue_mask, depth_map, focus_ratio=1.0
    ):
        """Fast ROI extraction with simplified logic"""
        H, W = image.shape[:2]

        # Resize masks
        tool_mask_resized = cv2.resize(
            tool_mask.astype(np.uint8), (W, H), interpolation=cv2.INTER_NEAREST
        )
        tissue_mask_resized = cv2.resize(
            tissue_mask.astype(np.uint8), (W, H), interpolation=cv2.INTER_NEAREST
        )

        # Get union mask (simpler approach)
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

    def _create_tti_result(
        self, pair, bbox, frame_idx, tti_class, tti_confidence, frame_shape
    ):
        """Create TTI result object with separate bounding boxes for tissue and tool"""
        if tti_class == 1:
            tti_name = TTI_CLASS_POSITIVE_NAME
            tti_value = TTI_CLASS_POSITIVE_VALUE
            color = TTI_CLASS_POSITIVE_COLOR
        else:
            tti_name = TTI_CLASS_NEGATIVE_NAME
            tti_value = TTI_CLASS_NEGATIVE_VALUE
            color = TTI_CLASS_NEGATIVE_COLOR

        # Calculate tissue interaction bounding box
        tissue_mask = pair["tissue"]["mask"]
        H_full, W_full = frame_shape[:2]
        tissue_mask_resized = cv2.resize(
            tissue_mask.astype(np.uint8),
            (W_full, H_full),
            interpolation=cv2.INTER_NEAREST,
        )
        if cv2.countNonZero(tissue_mask_resized) > 0:
            tissue_x, tissue_y, tissue_w, tissue_h = cv2.boundingRect(
                tissue_mask_resized
            )
            tti_bounding_box = {
                "x": float(tissue_x) / W_full,
                "y": float(tissue_y) / H_full,
                "w": float(tissue_w) / W_full,
                "h": float(tissue_h) / H_full,
            }
        else:
            tti_bounding_box = {"x": 0.0, "y": 0.0, "w": 0.0, "h": 0.0}

        # Calculate tool bounding box
        tool_mask = pair["tool"]["mask"]
        tool_mask_resized = cv2.resize(
            tool_mask.astype(np.uint8),
            (W_full, H_full),
            interpolation=cv2.INTER_NEAREST,
        )
        if cv2.countNonZero(tool_mask_resized) > 0:
            tool_x, tool_y, tool_w, tool_h = cv2.boundingRect(tool_mask_resized)
            tool_bounding_box = {
                "x": float(tool_x) / W_full,
                "y": float(tool_y) / H_full,
                "w": float(tool_w) / W_full,
                "h": float(tool_h) / H_full,
            }
        else:
            tool_bounding_box = {"x": 0.0, "y": 0.0, "w": 0.0, "h": 0.0}

        return {
            "featureHash": str(uuid.uuid4())[:8],
            "objectHash": str(uuid.uuid4())[:8],
            "name": tti_name,
            "value": tti_value,
            "color": color,
            "shape": "bounding_box",
            "confidence": float(tti_confidence),
            "frame": frame_idx,
            "createdBy": "Optimized_ViT_TTI_Classifier",
            "createdAt": datetime.now().strftime("%a, %d %b %Y %H:%M:%S GMT"),
            "lastEditedBy": "Optimized_ViT_TTI_Classifier",
            "lastEditedAt": datetime.now().strftime("%a, %d %b %Y %H:%M:%S GMT"),
            "manualAnnotation": False,
            "boundingBox": {
                "x": float(bbox[0]) / frame_shape[1],
                "y": float(bbox[1]) / frame_shape[0],
                "w": float(bbox[2]) / frame_shape[1],
                "h": float(bbox[3]) / frame_shape[0],
            },
            "tti_bounding_box": tti_bounding_box,
            "tool_bounding_box": tool_bounding_box,
            "tool_info": {
                "class": pair["tool"]["class"],
                "name": self.TOOL_NAMES.get(
                    pair["tool"]["class"], f"tool_{pair['tool']['class']}"
                ),
                "confidence": float(pair["tool"]["confidence"]),
            },
            "tissue_info": {
                "class": pair["tissue"]["class"],
                "name": self.TTI_NAMES.get(
                    pair["tissue"]["class"], f"tti_{pair['tissue']['class']}"
                ),
                "confidence": float(pair["tissue"]["confidence"]),
            },
            "tti_classification": tti_class,
        }

    def _deduplicate_final_results_fast(self, results, iou_threshold=0.3):
        """Fast deduplication with simplified logic"""
        if len(results) <= 1:
            return results

        # Sort by confidence
        results = sorted(results, key=lambda r: r["confidence"], reverse=True)

        deduplicated = []
        for current in results:
            is_duplicate = False
            current_bbox = current["boundingBox"]

            for existing in deduplicated:
                existing_bbox = existing["boundingBox"]

                # Simple center distance check
                current_center = (
                    current_bbox["x"] + current_bbox["w"] / 2,
                    current_bbox["y"] + current_bbox["h"] / 2,
                )
                existing_center = (
                    existing_bbox["x"] + existing_bbox["w"] / 2,
                    existing_bbox["y"] + existing_bbox["h"] / 2,
                )

                distance = (
                    (current_center[0] - existing_center[0]) ** 2
                    + (current_center[1] - existing_center[1]) ** 2
                ) ** 0.5

                if distance < 0.1:  # Simple distance threshold
                    is_duplicate = True
                    break

            if not is_duplicate:
                deduplicated.append(current)

        return deduplicated

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

    def _convert_to_json_serializable(self, obj):
        """Convert numpy types to JSON serializable types"""
        if isinstance(obj, dict):
            return {
                key: self._convert_to_json_serializable(value)
                for key, value in obj.items()
            }
        elif isinstance(obj, list):
            return [self._convert_to_json_serializable(item) for item in obj]
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif hasattr(obj, "item"):  # For scalar numpy types
            return obj.item()
        else:
            return obj

    def evaluate_video(
        self,
        video_path,
        output_path,
        output_video_path=None,
        start_frame=0,
        end_frame=None,
        frame_step=1,
        show_heatmap=True,
        show_roi_box=False,
    ):
        """Evaluate TTI detection on video with batch processing"""
        print(f"Starting optimized TTI evaluation on video: {video_path}")

        cap = cv2.VideoCapture(video_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        if end_frame is None:
            end_frame = total_frames

        print(f"Video info: {total_frames} frames, {fps} FPS, {width}x{height}")
        if frame_step == 1:
            print("Processing ALL frames for real-time playback")
        else:
            print(f"Processing every {frame_step} frames (faster processing)")
        print(f"Batch size: {self.batch_size}")

        if show_heatmap:
            print(
                "Heat map overlays: ENABLED (blue=tools, green=tissues, only for TTI detected)"
            )
        if show_roi_box:
            print("ROI bounding boxes: ENABLED")

        # Generate unique hashes
        label_hash = str(uuid.uuid4())
        dataset_hash = str(uuid.uuid4())
        data_hash = str(uuid.uuid4())

        # Initialize results structure
        results = [
            {
                "label_hash": label_hash,
                "branch_name": "main",
                "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "last_edited_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "dataset_hash": dataset_hash,
                "dataset_title": "Optimized ViT TTI Evaluation",
                "data_title": os.path.basename(video_path),
                "data_hash": data_hash,
                "data_type": "video",
                "is_image_sequence": None,
                "data_units": {
                    data_hash: {
                        "data_hash": data_hash,
                        "data_title": os.path.basename(video_path),
                        "data_type": "video/mp4",
                        "data_sequence": 0,
                        "labels": {},
                        "width": width,
                        "height": height,
                        "data_fps": fps,
                        "data_duration": total_frames / fps if fps > 0 else 0,
                        "data_link": video_path,
                    }
                },
                "object_answers": {},
            }
        ]

        # Initialize video writer
        video_writer = None
        if output_video_path:
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            video_writer = cv2.VideoWriter(
                output_video_path, fourcc, fps, (width, height)
            )

        # Process frames in batches
        frame_indices = list(
            range(start_frame, min(end_frame, total_frames), frame_step)
        )
        total_ttis_detected = 0
        frames_with_tti = 0

        for i in tqdm(
            range(0, len(frame_indices), self.batch_size), desc="Processing batches"
        ):
            batch_indices = frame_indices[i : i + self.batch_size]
            batch_frames = []

            # Load batch frames
            for frame_idx in batch_indices:
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
                ret, frame = cap.read()
                if ret:
                    batch_frames.append(frame)
                else:
                    print(f"Failed to read frame {frame_idx}")

            if not batch_frames:
                continue

            # Process batch
            batch_results, annotated_frames = self.process_frames_batch(
                batch_frames,
                batch_indices[: len(batch_frames)],
                draw_annotations=video_writer is not None,
                show_heatmap=show_heatmap,
                show_roi_box=show_roi_box,
            )

            # Add results and write video frames
            for j, (frame_result, annotated_frame) in enumerate(
                zip(batch_results, annotated_frames)
            ):
                frame_idx = batch_indices[j]

                # Convert frame_result to JSON serializable format
                frame_result = self._convert_to_json_serializable(frame_result)

                if frame_result["objects"]:
                    results[0]["data_units"][data_hash]["labels"][str(frame_idx)] = (
                        frame_result
                    )

                    tti_count = sum(
                        1
                        for obj in frame_result["objects"]
                        if obj.get("tti_classification") == 1
                    )
                    total_ttis_detected += tti_count
                    if tti_count > 0:
                        frames_with_tti += 1
                else:
                    results[0]["data_units"][data_hash]["labels"][str(frame_idx)] = {
                        "objects": [],
                        "classifications": [],
                    }

                if video_writer and annotated_frame is not None:
                    video_writer.write(annotated_frame)

        cap.release()
        if video_writer:
            video_writer.release()

        # Convert entire results to JSON serializable format before saving
        print("Converting results to JSON serializable format...")
        results = self._convert_to_json_serializable(results)

        # Save results
        print(f"Saving results to: {output_path}")
        try:
            with open(output_path, "w") as f:
                json.dump(results, f, indent=2)
            print("Results saved successfully!")
        except Exception as e:
            print(f"Error saving JSON results: {e}")
            backup_path = output_path.replace(".json", "_backup.pkl")
            import pickle

            with open(backup_path, "wb") as f:
                pickle.dump(results, f)
            print(f"Results saved as pickle backup: {backup_path}")

        # Print summary
        total_frames_processed = len(frame_indices)
        print("=" * 60)
        print("OPTIMIZED EVALUATION SUMMARY")
        print("=" * 60)
        print(f"Frames processed: {total_frames_processed}")
        print(f"Total TTIs detected: {total_ttis_detected}")
        print(f"Frames with TTI: {frames_with_tti}")
        if total_frames_processed > 0:
            print(
                f"TTI detection rate: {frames_with_tti / total_frames_processed * 100:.1f}%"
            )
        print(f"Results saved to: {output_path}")

    def evaluate_image(
        self,
        image_path,
        output_path,
        output_image_path=None,
        show_heatmap=True,
        show_roi_box=False,
    ):
        """Evaluate TTI detection on a single image."""
        print(f"Starting optimized TTI evaluation on image: {image_path}")

        frame = cv2.imread(image_path)
        if frame is None:
            print(f"Error: Could not read image file: {image_path}")
            return

        height, width, _ = frame.shape

        # Process the single frame as a batch of 1
        batch_results, annotated_frames = self.process_frames_batch(
            [frame],
            [0],  # frame index 0
            draw_annotations=True,
            show_heatmap=show_heatmap,
            show_roi_box=show_roi_box,
        )

        # --- Adapt result saving from evaluate_video ---
        label_hash = str(uuid.uuid4())
        dataset_hash = str(uuid.uuid4())
        data_hash = str(uuid.uuid4())

        # Convert results to be JSON serializable
        frame_result_json = (
            self._convert_to_json_serializable(batch_results[0])
            if batch_results
            else {"objects": [], "classifications": []}
        )

        results = [
            {
                "label_hash": label_hash,
                "branch_name": "main",
                "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "last_edited_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "dataset_hash": dataset_hash,
                "dataset_title": "Optimized ViT TTI Evaluation (Image)",
                "data_title": os.path.basename(image_path),
                "data_hash": data_hash,
                "data_type": "image",
                "is_image_sequence": None,
                "data_units": {
                    data_hash: {
                        "data_hash": data_hash,
                        "data_title": os.path.basename(image_path),
                        "data_type": "image/jpeg",  # Or png, etc.
                        "labels": {"0": frame_result_json},
                        "width": width,
                        "height": height,
                    }
                },
                "object_answers": {},
            }
        ]
        # --- End of result adaptation ---

        # Save JSON results
        print(f"Saving results to: {output_path}")
        try:
            with open(output_path, "w") as f:
                json.dump(results, f, indent=2)
            print("Results saved successfully!")
        except Exception as e:
            print(f"Error saving JSON results: {e}")

        # Save annotated image
        if output_image_path and annotated_frames and annotated_frames[0] is not None:
            cv2.imwrite(output_image_path, annotated_frames[0])
            print(f"Annotated image saved to: {output_image_path}")

        # Print summary
        total_ttis_detected = 0
        if frame_result_json and frame_result_json["objects"]:
            total_ttis_detected = sum(
                1
                for obj in frame_result_json["objects"]
                if obj.get("tti_classification") == 1
            )

        print("=" * 60)
        print("OPTIMIZED EVALUATION SUMMARY (IMAGE)")
        print("=" * 60)
        print(f"Total TTIs detected: {total_ttis_detected}")
        print(f"Results saved to: {output_path}")
        if output_image_path:
            print(f"Annotated image saved to: {output_image_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Optimized ViT TTI Classifier evaluation with detailed visualization control"
    )
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument("--video", help="Path to input video file")
    input_group.add_argument("--image", help="Path to input image file")

    parser.add_argument("--output", required=True, help="Path to output JSON file")
    parser.add_argument(
        "--output_video",
        default=None,
        help="Path to save annotated video (if --video is used)",
    )
    parser.add_argument(
        "--output_image",
        default=None,
        help="Path to save annotated image (if --image is used)",
    )
    parser.add_argument(
        "--show_heatmap",
        action="store_true",
        help="Show heat map overlays (blue=tools, green=tissues)",
    )
    parser.add_argument(
        "--show_roi_box",
        action="store_true",
        help="Show ROI bounding boxes (default: False)",
    )
    parser.add_argument(
        "--vit_model", default=DEFAULT_VIT_MODEL_PATH, help="Path to ViT model"
    )
    parser.add_argument(
        "--yolo_model", default=DEFAULT_YOLO_MODEL_PATH, help="Path to YOLO model"
    )
    parser.add_argument(
        "--start_frame", type=int, default=0, help="Starting frame index (for video)"
    )
    parser.add_argument(
        "--end_frame", type=int, default=None, help="Ending frame index (for video)"
    )
    parser.add_argument(
        "--frame_step",
        type=int,
        default=DEFAULT_FRAME_STEP,
        help="Frame step size (for video)",
    )
    parser.add_argument("--device", default=None, help="Device to use (cuda/cpu/mps)")
    parser.add_argument(
        "--focus_ratio",
        type=float,
        default=DEFAULT_FOCUS_RATIO,
        help="Focus ratio for ROI extraction",
    )
    parser.add_argument(
        "--depth_model", type=str, default=None, help="Path to local depth model"
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=DEFAULT_BATCH_SIZE,
        help="Batch size for processing",
    )
    parser.add_argument(
        "--disable_half_precision",
        action="store_true",
        help="Disable half precision inference",
    )

    args = parser.parse_args()

    # Validate inputs
    if args.video and not os.path.exists(args.video):
        print(f"Error: Video file not found: {args.video}")
        return
    if args.image and not os.path.exists(args.image):
        print(f"Error: Image file not found: {args.image}")
        return

    if not os.path.exists(args.vit_model):
        print(f"Error: ViT model not found: {args.vit_model}")
        return

    # Create output directory
    output_dir = os.path.dirname(args.output)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    # Initialize optimized evaluator
    evaluator = OptimizedTTIVideoEvaluator(
        vit_model_path=args.vit_model,
        yolo_model_path=args.yolo_model,
        device=args.device,
        focus_ratio=args.focus_ratio,
        depth_model_path=args.depth_model,
        batch_size=args.batch_size,
        use_half_precision=not args.disable_half_precision,
    )

    if args.video:
        # Run evaluation on video
        evaluator.evaluate_video(
            video_path=args.video,
            output_path=args.output,
            output_video_path=args.output_video,
            start_frame=args.start_frame,
            end_frame=args.end_frame,
            frame_step=args.frame_step,
            show_heatmap=args.show_heatmap,
            show_roi_box=args.show_roi_box,
        )
    elif args.image:
        # Determine output image path
        output_image_path = args.output_image
        if not output_image_path and args.output:
            # Default output path next to the json
            base, _ = os.path.splitext(args.output)
            output_image_path = base + "_annotated.png"

        # Run evaluation on image
        evaluator.evaluate_image(
            image_path=args.image,
            output_path=args.output,
            output_image_path=output_image_path,
            show_heatmap=args.show_heatmap,
            show_roi_box=args.show_roi_box,
        )


if __name__ == "__main__":
    main()
