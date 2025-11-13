#!/usr/bin/env python3
"""
Interaction Area Inference Script

Usage:
  python interaction_area_only_inference.py --video /path/to/video.mp4 [--output /path/to/results.json] [--visualize]

Arguments:
- --video: Path to the input video file to analyze.
- --output: Optional path to the output JSON file. Defaults to './<video_name>_interaction_areas.json'.
- --visualize: If set, outputs frames with interaction areas drawn.
- --output_frames_dir: Optional directory to save visualized frames. Defaults to './<video_name>_interaction_frames'.
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
DEFAULT_VIT_MODEL_PATH = "/cluster/projects/madanigroup/lorenz/tti/models/vit.pt"
DEFAULT_YOLO_MODEL_PATH = "/cluster/projects/madanigroup/lorenz/tti/models/yolo.pt"
DEFAULT_DEPTH_MODEL_PATH = (
    "/cluster/projects/madanigroup/lorenz/tti/models/dpt-large"
)

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
    """Optimized Video evaluator for TTI classification using batch processing."""

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
        if device:
            self.device = device
        else:
            if torch.cuda.is_available():
                self.device = "cuda"
            elif torch.backends.mps.is_available():
                self.device = "mps"
            else:
                self.device = "cpu"

        self.focus_ratio, self.batch_size = focus_ratio, batch_size
        self.TOOL_CLASSES, self.TTI_CLASSES = TOOL_CLASSES, TTI_CLASSES
        self.TOOL_NAMES, self.TTI_NAMES = TOOL_NAMES, TTI_NAMES

        if self.device == "cuda":
            self.use_half_precision_vit, self.use_half_precision_yolo = (
                use_half_precision,
                False,
            )
            print(
                "CUDA detected: Disabling half precision for YOLO to avoid dtype conflicts"
            )
        else:
            self.use_half_precision_vit, self.use_half_precision_yolo = False, False

        print(f"Using device: {self.device}, Batch size: {batch_size}")
        print(
            f"ViT half precision: {self.use_half_precision_vit}, YOLO half precision: {self.use_half_precision_yolo}"
        )

        if self.device == "cuda":
            torch.backends.cudnn.benchmark = True
            torch.backends.cudnn.deterministic = False

        self.tti_model = self._load_tti_model(vit_model_path)
        self.yolo_model = self._load_yolo_model(yolo_model_path)
        self._load_depth_model(depth_model_path)
        self.depth_cache = {}
        print("All models loaded successfully!")

    def _load_tti_model(self, model_path):
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
                features = self.backbone(pixel_values=x).last_hidden_state[:, 0]
                return F.sigmoid(self.fc(features))

        return OptimizedROIClassifierViT(num_hoi_classes)

    def _load_yolo_model(self, model_path):
        print(f"Loading YOLO model from: {model_path}")
        model = YOLO(model_path)
        if self.device == "cuda":
            try:
                model.model.float()
                print("YOLO model forced to float32 for CUDA stability")
            except Exception as e:
                print(f"Warning: Could not force YOLO to float32: {e}")
        return model

    def _load_depth_model(self, depth_model_path):
        print("Loading depth estimation model...")
        if depth_model_path and os.path.exists(depth_model_path):
            try:
                device_id = 0 if self.device in ["cuda", "mps"] else -1
                self.depth_model = pipeline(
                    task="depth-estimation", model=depth_model_path, device=device_id
                )
                self.use_real_depth = True
                print("Depth model loaded successfully from local path!")
            except Exception as e:
                print(f"Failed to load local depth model at {depth_model_path}: {e}")
                print("Using fallback depth estimation.")
                self.depth_model, self.use_real_depth = None, False
        else:
            if depth_model_path:
                print(f"Local depth model path not found: {depth_model_path}")
            else:
                print("No local depth model path provided.")
            print("Using fallback depth estimation. To use a real depth model,")
            print(
                "download 'Intel/dpt-large' from Hugging Face and provide the path via --depth_model."
            )
            self.depth_model, self.use_real_depth = None, False

    def _fallback_depth_estimation(self, image):
        img_array = np.array(image)
        gray = (
            cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
            if len(img_array.shape) == 3
            else img_array
        )
        edges = cv2.Canny(gray, 50, 150)
        dist_transform = cv2.distanceTransform(255 - edges, cv2.DIST_L2, 5)
        return 255 - cv2.normalize(
            dist_transform, None, 0, 255, cv2.NORM_MINMAX
        ).astype(np.uint8)

    def process_frames_batch(
        self,
        frames: List[np.ndarray],
        frame_indices: List[int],
        visualize: bool = False,
    ) -> Tuple[List[Dict], List[np.ndarray]]:
        batch_results, annotated_frames = [], (
            [frame.copy() for frame in frames] if visualize else []
        )
        try:
            yolo_results_batch = self.yolo_model(frames, verbose=False)
        except Exception as e:
            print(
                f"Batch YOLO inference failed: {e}, falling back to single frame processing."
            )
            yolo_results_batch = [
                self.yolo_model(frame, verbose=False)[0] for frame in frames
            ]

        for i, (frame, frame_idx) in enumerate(zip(frames, frame_indices)):
            if i >= len(yolo_results_batch) or yolo_results_batch[i] is None:
                batch_results.append({"frame": frame_idx, "objects": []})
                continue

            depth_cache_key = f"{frame_idx}"
            if depth_cache_key in self.depth_cache:
                depth_map = self.depth_cache[depth_cache_key]
            else:
                pil_frame = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                if self.use_real_depth and self.depth_model:
                    try:
                        depth_map = np.array(self.depth_model(pil_frame)["depth"])
                    except Exception:
                        depth_map = self._fallback_depth_estimation(pil_frame)
                else:
                    depth_map = self._fallback_depth_estimation(pil_frame)
                self.depth_cache[depth_cache_key] = depth_map
                if len(self.depth_cache) > DEPTH_CACHE_MAX_SIZE:
                    del self.depth_cache[min(self.depth_cache.keys())]

            detections = self.parse_yolo_output([yolo_results_batch[i]])
            pairs = self.find_tool_tissue_pairs_fast(detections)
            frame_result = {"frame": frame_idx, "objects": []}
            roi_batch, pair_info = [], []
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
                    if self.use_half_precision_vit:
                        roi_tensor = roi_tensor.half()
                    roi_batch.append(roi_tensor)
                    pair_info.append((pair, bbox, frame_idx))

            if roi_batch:
                try:
                    roi_batch_tensor = torch.stack(roi_batch).to(self.device)
                    with torch.no_grad():
                        logits_batch = self.tti_model(roi_batch_tensor)
                        tti_classes = torch.argmax(logits_batch, dim=1).cpu().numpy()
                        tti_confidences = (
                            F.softmax(logits_batch, dim=1).max(dim=1)[0].cpu().numpy()
                        )
                    for j, (pair, bbox, frame_idx) in enumerate(pair_info):
                        tti_result = self._create_tti_result(
                            pair,
                            bbox,
                            frame_idx,
                            tti_classes[j],
                            tti_confidences[j],
                            frame.shape,
                        )
                        if tti_result:
                            frame_result["objects"].append(tti_result)
                except Exception as e:
                    print(f"ViT inference failed for frame {frame_idx}: {e}")

            if frame_result["objects"]:
                frame_result["objects"] = self._deduplicate_final_results_fast(
                    frame_result["objects"]
                )

            if visualize:
                H_full, W_full = frame.shape[:2]
                combined_tissue_mask = np.zeros((H_full, W_full), dtype=np.uint8)
                for obj in frame_result["objects"]:
                    if obj.get("tti_classification") == 1:
                        for polygon_obj in obj["interaction_polygons"]:
                            polygon_coords = polygon_obj["coordinates"][0]
                            contour = (
                                np.array(polygon_coords) * [W_full, H_full]
                            ).astype(np.int32)
                            cv2.fillPoly(combined_tissue_mask, [contour], 1)
                if np.any(combined_tissue_mask):
                    annotated_frames[i] = show_mask_overlay_from_binary_mask(
                        annotated_frames[i],
                        combined_tissue_mask,
                        alpha=HEATMAP_ALPHA,
                        mask_color=HEATMAP_TISSUE_COLOR,
                    )
            batch_results.append(frame_result)
        return batch_results, annotated_frames

    def find_tool_tissue_pairs_fast(self, detections):
        tools = [d for d in detections if d["type"] == "tool"]
        tissues = [d for d in detections if d["type"] == "tti"]
        pairs = [
            {"tool": tool, "tissue": tissue} for tool in tools for tissue in tissues
        ]
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
        H, W = image.shape[:2]
        tool_mask_resized = cv2.resize(
            tool_mask.astype(np.uint8), (W, H), interpolation=cv2.INTER_NEAREST
        )
        tissue_mask_resized = cv2.resize(
            tissue_mask.astype(np.uint8), (W, H), interpolation=cv2.INTER_NEAREST
        )
        union_mask = cv2.bitwise_or(tool_mask_resized, tissue_mask_resized)
        if cv2.countNonZero(union_mask) == 0:
            return None, None
        x, y, w, h = cv2.boundingRect(union_mask)
        w, h = max(w, ROI_MIN_SIZE), max(h, ROI_MIN_SIZE)
        x, y = max(0, min(x, W - w)), max(0, min(y, H - h))
        w, h = min(w, W - x), min(h, H - y)
        roi = np.concatenate(
            [
                image[y : y + h, x : x + w],
                depth_map[y : y + h, x : x + w][..., None],
                union_mask[y : y + h, x : x + w][..., None],
            ],
            axis=-1,
        )
        return roi, (x, y, w, h)

    def _create_tti_result(
        self, pair, bbox, frame_idx, tti_class, tti_confidence, frame_shape
    ):
        H_full, W_full = frame_shape[:2]
        tissue_mask_resized = cv2.resize(
            pair["tissue"]["mask"].astype(np.uint8),
            (W_full, H_full),
            interpolation=cv2.INTER_NEAREST,
        )

        interaction_polygons = []
        if cv2.countNonZero(tissue_mask_resized) > 0:
            contours, _ = cv2.findContours(
                tissue_mask_resized, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
            )
            for contour in contours:
                if contour.shape[0] > 2:
                    contour_points = contour.squeeze(axis=1).astype(np.float32)
                    contour_points[:, 0] /= W_full
                    contour_points[:, 1] /= H_full
                    geojson_polygon = {
                        "type": "Polygon",
                        "coordinates": [contour_points.tolist()],
                    }
                    interaction_polygons.append(geojson_polygon)

        return {
            "frame": frame_idx,
            "interaction_polygons": interaction_polygons,
            "tti_classification": int(tti_class),
            "confidence": float(tti_confidence),
        }

    def _deduplicate_final_results_fast(self, results, iou_threshold=0.3):
        if len(results) <= 1:
            return results
        return sorted(results, key=lambda r: r["confidence"], reverse=True)

    def parse_yolo_output(self, result):
        if (
            not result
            or not result[0]
            or result[0].boxes is None
            or result[0].masks is None
        ):
            return []
        r = result[0]
        classes, masks, boxes, confs = (
            r.boxes.cls.cpu().numpy(),
            r.masks.data.cpu().numpy(),
            r.boxes.data.cpu().numpy(),
            r.boxes.conf.cpu().numpy(),
        )
        return [
            {
                "class": int(cls),
                "mask": mask,
                "box": box,
                "confidence": float(conf),
                "type": "tool" if int(cls) in self.TOOL_CLASSES else "tti",
            }
            for cls, mask, box, conf in zip(classes, masks, boxes, confs)
        ]

    def _convert_to_json_serializable(self, obj):
        if isinstance(obj, dict):
            return {k: self._convert_to_json_serializable(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [self._convert_to_json_serializable(i) for i in obj]
        if isinstance(obj, (np.integer, np.floating)):
            return obj.item()
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return obj

    def evaluate_video(
        self,
        video_path,
        output_path,
        start_frame=0,
        end_frame=None,
        frame_step=1,
        visualize=False,
        output_frames_dir=None,
    ):
        print(f"Starting interaction area inference on video: {video_path}")
        cap = cv2.VideoCapture(video_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        end_frame = end_frame or total_frames
        results = {"video_name": os.path.basename(video_path)}
        frame_indices = list(
            range(start_frame, min(end_frame, total_frames), frame_step)
        )

        if visualize and output_frames_dir:
            os.makedirs(output_frames_dir, exist_ok=True)
            print(f"Visualized frames will be saved to: {output_frames_dir}")

        for i in tqdm(
            range(0, len(frame_indices), self.batch_size), desc="Processing batches"
        ):
            batch_indices = frame_indices[i : i + self.batch_size]
            batch_frames, read_indices = [], []
            for frame_idx in batch_indices:
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
                ret, frame = cap.read()
                if ret:
                    batch_frames.append(frame)
                    read_indices.append(frame_idx)
                else:
                    results[str(frame_idx)] = []

            if not batch_frames:
                continue

            batch_results, annotated_frames = self.process_frames_batch(
                batch_frames, read_indices, visualize=visualize
            )

            processed_in_batch = {res["frame"] for res in batch_results}
            for res in batch_results:
                frame_polygons = []
                for o in res.get("objects", []):
                    if o.get("tti_classification") == 1:
                        frame_polygons.extend(o["interaction_polygons"])
                results[str(res["frame"])] = frame_polygons

            for idx in read_indices:
                if idx not in processed_in_batch:
                    results[str(idx)] = []

            if visualize:
                for frame_idx, annotated_frame in zip(read_indices, annotated_frames):
                    if results.get(str(frame_idx)):
                        cv2.imwrite(
                            os.path.join(output_frames_dir, f"frame_{frame_idx}.jpg"),
                            annotated_frame,
                        )

        cap.release()
        results = self._convert_to_json_serializable(results)
        with open(output_path, "w") as f:
            json.dump(results, f, indent=2)
        print("Results saved successfully!")

        total_interactions = sum(len(v) for k, v in results.items() if k.isdigit())
        frames_with_interaction = sum(
            1 for k, v in results.items() if k.isdigit() and v
        )
        print("=" * 60 + f"\nINTERACTION AREA INFERENCE SUMMARY\n" + "=" * 60)
        print(f"Frames processed: {len(frame_indices)}")
        print(f"Total interaction areas found: {total_interactions}")
        print(f"Frames with interaction: {frames_with_interaction}")
        print(f"Results saved to: {output_path}")
        if visualize:
            print(f"Visualized frames saved to: {output_frames_dir}")


def main():
    parser = argparse.ArgumentParser(
        description="Extracts interaction areas (green segmentations) from a video."
    )
    parser.add_argument("--video", required=True, help="Path to input video file")
    parser.add_argument(
        "--output",
        default="/cluster/projects/madanigroup/lorenz/tti/interaction-areas/",
        help="Path to output JSON file. Defaults to './<video_name>_interaction_areas.json'",
    )
    parser.add_argument(
        "--visualize",
        action="store_true",
        help="Output frames with interaction areas drawn.",
    )
    parser.add_argument(
        "--output_frames_dir", default=None, help="Directory to save visualized frames."
    )
    parser.add_argument(
        "--vit_model", default=DEFAULT_VIT_MODEL_PATH, help="Path to ViT model"
    )
    parser.add_argument(
        "--yolo_model", default=DEFAULT_YOLO_MODEL_PATH, help="Path to YOLO model"
    )
    parser.add_argument(
        "--start_frame", type=int, default=0, help="Starting frame index"
    )
    parser.add_argument(
        "--end_frame", type=int, default=None, help="Ending frame index"
    )
    parser.add_argument(
        "--frame_step", type=int, default=DEFAULT_FRAME_STEP, help="Frame step size"
    )
    parser.add_argument("--device", default=None, help="Device to use (cuda/cpu/mps)")
    parser.add_argument(
        "--focus_ratio",
        type=float,
        default=DEFAULT_FOCUS_RATIO,
        help="Focus ratio for ROI extraction",
    )
    parser.add_argument(
        "--depth_model",
        type=str,
        default=DEFAULT_DEPTH_MODEL_PATH,
        help="Path to local depth model",
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

    if not os.path.exists(args.video):
        print(f"Error: Video file not found: {args.video}")
        return
    if not os.path.exists(args.vit_model):
        print(f"Error: ViT model not found: {args.vit_model}")
        return

    video_basename = os.path.splitext(os.path.basename(args.video))[0]
    output_path = args.output or f"./{video_basename}_interaction_areas.json"
    output_frames_dir = args.output_frames_dir
    if args.visualize and not output_frames_dir:
        output_frames_dir = f"./{video_basename}_interaction_frames"

    if os.path.dirname(output_path):
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

    evaluator = OptimizedTTIVideoEvaluator(
        vit_model_path=args.vit_model,
        yolo_model_path=args.yolo_model,
        device=args.device,
        focus_ratio=args.focus_ratio,
        depth_model_path=args.depth_model,
        batch_size=args.batch_size,
        use_half_precision=not args.disable_half_precision,
    )
    evaluator.evaluate_video(
        video_path=args.video,
        output_path=output_path,
        start_frame=args.start_frame,
        end_frame=args.end_frame,
        frame_step=args.frame_step,
        visualize=args.visualize,
        output_frames_dir=output_frames_dir,
    )


if __name__ == "__main__":
    main()
