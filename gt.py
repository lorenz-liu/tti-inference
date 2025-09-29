#!/usr/bin/env python3
"""
Project I: Enhanced IoU/Dice/GT Coverage Analysis for TTI Model Validation
Compares TTI model predictions (bounding boxes) with ground truth bounding box annotations
Includes GT Coverage Ratio as a more practical metric for TTI detection
"""

import json
import os
import glob
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
import cv2
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
import argparse


class EnhancedTTIValidationAnalyzer:
    def __init__(
        self,
        tti_output_dir: str,
        ground_truth_dir: str,
        videos_dir: str,
        output_prefix: str = "enhanced_project_i",
    ):
        """
        Initialize the enhanced TTI validation analyzer

        Args:
            tti_output_dir: Path to inference-by-lorenz directory
            ground_truth_dir: Path to ground truth annotations directory
            videos_dir: Path to videos directory
            output_prefix: Prefix for output files (default: "enhanced_project_i")
        """
        self.tti_output_dir = Path(tti_output_dir)
        self.ground_truth_dir = Path(ground_truth_dir)
        self.videos_dir = Path(videos_dir)
        self.output_prefix = output_prefix

        # Storage for results
        self.file_mapping = {}
        self.results = []

    def create_file_mapping(self):
        """Create mapping between ground truth UUIDs and TTI output files"""
        print("Creating file mapping...")

        # Get all ground truth files
        gt_files = glob.glob(str(self.ground_truth_dir / "*.json"))

        for gt_file in gt_files:
            with open(gt_file, "r") as f:
                gt_data = json.load(f)

            # Extract video title from ground truth - handle the nested structure
            if isinstance(gt_data, list) and len(gt_data) > 0:
                gt_entry = gt_data[0]
                video_title = gt_entry.get("data_title", "")
                uuid = gt_entry.get("data_hash", "")

                if not video_title or not uuid:
                    print(f"Warning: Missing data_title or data_hash in {gt_file}")
                    continue

                print(f"Processing: {video_title} (UUID: {uuid})")

                # Determine if safe or bdi based on video title
                video_type = "safe" if "LapChol" in video_title else "bdi"

                # Create normalized name for matching
                if "LapChol" in video_title:
                    # Convert "LapChol Case 0015 01.mp4" to "lapchol_0015_01"
                    # Remove file extension first
                    clean_title = video_title.replace(".mp4", "").replace(".MP4", "")
                    parts = clean_title.split()
                    if len(parts) >= 4:
                        # Extract case number and part number
                        case_num = parts[2]  # "0015"
                        part_num = parts[3]  # "01"
                        normalized_name = f"lapchol_{case_num}_{part_num}"
                    else:
                        print(f"Warning: Unexpected LapChol format: {video_title}")
                        normalized_name = clean_title.replace(" ", "_").lower()
                else:
                    # Handle BDI cases - extract V number
                    # "V10-Trimmed.mp4" -> "v10", "V2_Trimmed.mp4" -> "v2"
                    if video_title.startswith("V"):
                        # Extract the number after V
                        import re

                        match = re.match(r"V(\d+)", video_title)
                        if match:
                            v_number = match.group(1)
                            normalized_name = f"v{v_number}"
                        else:
                            normalized_name = (
                                video_title.replace(".mp4", "")
                                .replace(".MP4", "")
                                .lower()
                            )
                    else:
                        normalized_name = (
                            video_title.replace(".mp4", "").replace(".MP4", "").lower()
                        )

                print(f"  Normalized name: {normalized_name}")

                # Find corresponding TTI output file (flat directory structure)
                tti_pattern = str(
                    self.tti_output_dir / f"*{normalized_name}*_vit_eval.json"
                )
                tti_files = glob.glob(tti_pattern)

                # If no exact match, try looser matching
                if not tti_files:
                    # For LapChol, try with different case combinations
                    if "LapChol" in video_title:
                        case_part = normalized_name.replace("lapchol_", "")
                        alt_patterns = [
                            f"*{case_part}*_vit_eval.json",
                            f"*lapchol*{case_part}*_vit_eval.json",
                        ]
                        for pattern in alt_patterns:
                            tti_files = glob.glob(str(self.tti_output_dir / pattern))
                            if tti_files:
                                break
                    # For BDI, try alternative patterns
                    else:
                        v_num = normalized_name.replace("v", "")
                        alt_patterns = [
                            f"*v{v_num}*_vit_eval.json",
                            f"*V{v_num}*_vit_eval.json",
                        ]
                        for pattern in alt_patterns:
                            tti_files = glob.glob(str(self.tti_output_dir / pattern))
                            if tti_files:
                                break

                if tti_files:
                    self.file_mapping[uuid] = {
                        "ground_truth": gt_file,
                        "tti_output": tti_files[0],
                        "video_title": video_title,
                        "video_type": video_type,
                        "normalized_name": normalized_name,
                    }
                    print(f"  ✓ Mapped to: {os.path.basename(tti_files[0])}")
                else:
                    print(f"  ✗ No TTI output found for {video_title}")
                    print(f"    Tried pattern: {tti_pattern}")

                    # List available files for debugging
                    available_files = glob.glob(
                        str(self.tti_output_dir / "*_vit_eval.json")
                    )
                    if available_files:
                        print(f"    Available files in TTI output directory:")
                        for f in available_files:
                            print(f"      - {os.path.basename(f)}")

        print(f"\nSuccessfully mapped {len(self.file_mapping)} files")

        # Print mapping summary
        print("\nFILE MAPPING SUMMARY:")
        for uuid, info in self.file_mapping.items():
            print(f"  {info['video_title']} -> {os.path.basename(info['tti_output'])}")

        return self.file_mapping

    def load_ground_truth_annotations(self, gt_file: str) -> Dict:
        """Load and parse ground truth annotations"""
        with open(gt_file, "r") as f:
            gt_data = json.load(f)

        if isinstance(gt_data, list) and len(gt_data) > 0:
            gt_entry = gt_data[0]

            # Get video dimensions and title from the main entry
            width = 1280  # default
            height = 720  # default
            video_title = gt_entry.get("data_title", "")

            # Get data units
            data_units = gt_entry.get("data_units", {})

            # Extract labels from the data unit
            labels = {}
            for data_hash, data_unit in data_units.items():
                labels = data_unit.get("labels", {})
                width = data_unit.get("width", width)
                height = data_unit.get("height", height)
                break  # Take the first (and usually only) data unit

            return {
                "labels": labels,
                "width": width,
                "height": height,
                "video_title": video_title,
            }

        return {"labels": {}, "width": 1280, "height": 720, "video_title": ""}

    def load_tti_predictions(self, tti_file: str) -> Dict:
        """Load TTI model predictions"""
        with open(tti_file, "r") as f:
            tti_data = json.load(f)

        if isinstance(tti_data, list) and len(tti_data) > 0:
            tti_entry = tti_data[0]
            data_units = tti_entry.get("data_units", {})

            # Get the video data
            for data_hash, data_unit in data_units.items():
                labels = data_unit.get("labels", {})
                width = data_unit.get("width", 1280)
                height = data_unit.get("height", 720)

                return {"labels": labels, "width": width, "height": height}
        return {"labels": {}, "width": 1280, "height": 720}

    def calculate_bbox_iou(self, bbox1: Dict, bbox2: Dict) -> float:
        """Calculate IoU directly between two bounding boxes (normalized coordinates)"""
        # Get bounding box coordinates (already normalized)
        x1_min, y1_min = bbox1["x"], bbox1["y"]
        x1_max, y1_max = x1_min + bbox1["w"], y1_min + bbox1["h"]

        x2_min, y2_min = bbox2["x"], bbox2["y"]
        x2_max, y2_max = x2_min + bbox2["w"], y2_min + bbox2["h"]

        # Calculate intersection
        intersect_x_min = max(x1_min, x2_min)
        intersect_y_min = max(y1_min, y2_min)
        intersect_x_max = min(x1_max, x2_max)
        intersect_y_max = min(y1_max, y2_max)

        # Check if there is intersection
        if intersect_x_max <= intersect_x_min or intersect_y_max <= intersect_y_min:
            return 0.0

        # Calculate intersection area
        intersect_area = (intersect_x_max - intersect_x_min) * (
            intersect_y_max - intersect_y_min
        )

        # Calculate union area
        area1 = bbox1["w"] * bbox1["h"]
        area2 = bbox2["w"] * bbox2["h"]
        union_area = area1 + area2 - intersect_area

        # Avoid division by zero
        if union_area <= 0:
            return 0.0

        return intersect_area / union_area

    def calculate_bbox_dice(self, bbox1: Dict, bbox2: Dict) -> float:
        """Calculate DICE coefficient directly between two bounding boxes (normalized coordinates)"""
        # Get bounding box coordinates (already normalized)
        x1_min, y1_min = bbox1["x"], bbox1["y"]
        x1_max, y1_max = x1_min + bbox1["w"], y1_min + bbox1["h"]

        x2_min, y2_min = bbox2["x"], bbox2["y"]
        x2_max, y2_max = x2_min + bbox2["w"], y2_min + bbox2["h"]

        # Calculate intersection
        intersect_x_min = max(x1_min, x2_min)
        intersect_y_min = max(y1_min, y2_min)
        intersect_x_max = min(x1_max, x2_max)
        intersect_y_max = min(y1_max, y2_max)

        # Check if there is intersection
        if intersect_x_max <= intersect_x_min or intersect_y_max <= intersect_y_min:
            return 0.0

        # Calculate intersection area
        intersect_area = (intersect_x_max - intersect_x_min) * (
            intersect_y_max - intersect_y_min
        )

        # Calculate areas
        area1 = bbox1["w"] * bbox1["h"]
        area2 = bbox2["w"] * bbox2["h"]

        # Calculate DICE coefficient
        if area1 + area2 <= 0:
            return 0.0

        return 2 * intersect_area / (area1 + area2)

    def calculate_gt_coverage_ratio(self, pred_bbox: Dict, gt_bbox: Dict) -> float:
        """
        Calculate GT Coverage Ratio: How much of the ground truth is covered by prediction

        GT Coverage Ratio = (Intersection Area) / (GT Area)

        This is more practical for TTI detection as it answers:
        "Did we detect the interaction area that actually exists?"
        """
        # Get bounding box coordinates (already normalized)
        pred_x_min, pred_y_min = pred_bbox["x"], pred_bbox["y"]
        pred_x_max, pred_y_max = (
            pred_x_min + pred_bbox["w"],
            pred_y_min + pred_bbox["h"],
        )

        gt_x_min, gt_y_min = gt_bbox["x"], gt_bbox["y"]
        gt_x_max, gt_y_max = gt_x_min + gt_bbox["w"], gt_y_min + gt_bbox["h"]

        # Calculate intersection
        intersect_x_min = max(pred_x_min, gt_x_min)
        intersect_y_min = max(pred_y_min, gt_y_min)
        intersect_x_max = min(pred_x_max, gt_x_max)
        intersect_y_max = min(pred_y_max, gt_y_max)

        # Check if there is intersection
        if intersect_x_max <= intersect_x_min or intersect_y_max <= intersect_y_min:
            return 0.0

        # Calculate intersection area and GT area
        intersect_area = (intersect_x_max - intersect_x_min) * (
            intersect_y_max - intersect_y_min
        )
        gt_area = gt_bbox["w"] * gt_bbox["h"]

        # Calculate GT coverage ratio
        if gt_area <= 0:
            return 0.0

        return intersect_area / gt_area

    def calculate_prediction_precision(self, pred_bbox: Dict, gt_bbox: Dict) -> float:
        """
        Calculate Prediction Precision: How much of the prediction is actually correct

        Prediction Precision = (Intersection Area) / (Prediction Area)

        This answers: "How much of what we predicted is actually correct?"
        """
        # Get bounding box coordinates (already normalized)
        pred_x_min, pred_y_min = pred_bbox["x"], pred_bbox["y"]
        pred_x_max, pred_y_max = (
            pred_x_min + pred_bbox["w"],
            pred_y_min + pred_bbox["h"],
        )

        gt_x_min, gt_y_min = gt_bbox["x"], gt_bbox["y"]
        gt_x_max, gt_y_max = gt_x_min + gt_bbox["w"], gt_y_min + gt_bbox["h"]

        # Calculate intersection
        intersect_x_min = max(pred_x_min, gt_x_min)
        intersect_y_min = max(pred_y_min, gt_y_min)
        intersect_x_max = min(pred_x_max, gt_x_max)
        intersect_y_max = min(pred_y_max, gt_y_max)

        # Check if there is intersection
        if intersect_x_max <= intersect_x_min or intersect_y_max <= intersect_y_min:
            return 0.0

        # Calculate intersection area and prediction area
        intersect_area = (intersect_x_max - intersect_x_min) * (
            intersect_y_max - intersect_y_min
        )
        pred_area = pred_bbox["w"] * pred_bbox["h"]

        # Calculate prediction precision
        if pred_area <= 0:
            return 0.0

        return intersect_area / pred_area

    def calculate_center_distance_accuracy(
        self, pred_bbox: Dict, gt_bbox: Dict
    ) -> float:
        """
        Calculate center distance accuracy

        Returns a value between 0 and 1, where 1 means perfect center alignment
        """
        # Calculate centers
        pred_center_x = pred_bbox["x"] + pred_bbox["w"] / 2
        pred_center_y = pred_bbox["y"] + pred_bbox["h"] / 2

        gt_center_x = gt_bbox["x"] + gt_bbox["w"] / 2
        gt_center_y = gt_bbox["y"] + gt_bbox["h"] / 2

        # Calculate Euclidean distance
        distance = np.sqrt(
            (pred_center_x - gt_center_x) ** 2 + (pred_center_y - gt_center_y) ** 2
        )

        # Normalize by GT diagonal length
        gt_diagonal = np.sqrt(gt_bbox["w"] ** 2 + gt_bbox["h"] ** 2)

        if gt_diagonal <= 0:
            return 0.0

        # Convert distance to accuracy (closer = higher accuracy)
        accuracy = max(0.0, 1.0 - distance / gt_diagonal)
        return accuracy

    def is_detection_successful(
        self, pred_bbox: Dict, gt_bbox: Dict, threshold: float = 0.1
    ) -> bool:
        """
        Determine if the detection is successful based on GT coverage threshold

        Args:
            pred_bbox: Predicted bounding box
            gt_bbox: Ground truth bounding box
            threshold: Minimum GT coverage ratio to consider detection successful

        Returns:
            True if detection is successful, False otherwise
        """
        gt_coverage = self.calculate_gt_coverage_ratio(pred_bbox, gt_bbox)
        return gt_coverage >= threshold

    def find_best_match(self, gt_objects: List, pred_objects: List) -> List[Tuple]:
        """Find best matching pairs between ground truth and predictions"""
        matches = []

        for gt_obj in gt_objects:
            if "boundingBox" not in gt_obj:
                continue

            best_iou = 0.0
            best_match = None

            for pred_obj in pred_objects:
                if "boundingBox" not in pred_obj:
                    continue

                iou = self.calculate_bbox_iou(
                    gt_obj["boundingBox"], pred_obj["boundingBox"]
                )
                if iou > best_iou:
                    best_iou = iou
                    best_match = pred_obj

            if best_match is not None:
                matches.append((gt_obj, best_match, best_iou))

        return matches

    def analyze_frame(
        self, frame_id: str, gt_frame_data: Dict, pred_frame_data: Dict
    ) -> Dict:
        """Analyze a single frame with enhanced metrics"""
        gt_objects = gt_frame_data.get("objects", [])
        pred_objects = pred_frame_data.get("objects", [])

        if not gt_objects and not pred_objects:
            return None

        # Find best matches
        matches = self.find_best_match(gt_objects, pred_objects)

        frame_results = {
            "frame_id": frame_id,
            "gt_count": len(gt_objects),
            "pred_count": len(pred_objects),
            "matches": len(matches),
            "iou_scores": [],
            "dice_scores": [],
            "gt_coverage_ratios": [],
            "prediction_precisions": [],
            "center_distance_accuracies": [],
            "detection_successes": [],
        }

        for gt_obj, pred_obj, iou in matches:
            # Calculate all metrics
            dice = self.calculate_bbox_dice(
                gt_obj["boundingBox"], pred_obj["boundingBox"]
            )
            gt_coverage = self.calculate_gt_coverage_ratio(
                pred_obj["boundingBox"], gt_obj["boundingBox"]
            )
            pred_precision = self.calculate_prediction_precision(
                pred_obj["boundingBox"], gt_obj["boundingBox"]
            )
            center_accuracy = self.calculate_center_distance_accuracy(
                pred_obj["boundingBox"], gt_obj["boundingBox"]
            )
            is_successful = self.is_detection_successful(
                pred_obj["boundingBox"], gt_obj["boundingBox"]
            )

            # Store results
            frame_results["iou_scores"].append(iou)
            frame_results["dice_scores"].append(dice)
            frame_results["gt_coverage_ratios"].append(gt_coverage)
            frame_results["prediction_precisions"].append(pred_precision)
            frame_results["center_distance_accuracies"].append(center_accuracy)
            frame_results["detection_successes"].append(is_successful)

        return frame_results

    def analyze_video_pair(self, uuid: str, file_info: Dict) -> Dict:
        """Analyze a single video pair (ground truth vs predictions)"""
        print(f"Analyzing {file_info['video_title']}...")

        # Load data
        gt_data = self.load_ground_truth_annotations(file_info["ground_truth"])
        pred_data = self.load_tti_predictions(file_info["tti_output"])

        width, height = gt_data["width"], gt_data["height"]

        video_results = {
            "uuid": uuid,
            "video_title": file_info["video_title"],
            "video_type": file_info["video_type"],
            "width": width,
            "height": height,
            "frame_results": [],
            "total_gt_objects": 0,
            "total_pred_objects": 0,
            "total_matches": 0,
            "avg_iou": 0.0,
            "avg_dice": 0.0,
            "avg_gt_coverage": 0.0,
            "avg_prediction_precision": 0.0,
            "avg_center_accuracy": 0.0,
            "detection_success_rate": 0.0,
        }

        # Get common frame IDs
        gt_frames = set(gt_data["labels"].keys())
        pred_frames = set(pred_data["labels"].keys())
        common_frames = gt_frames.intersection(pred_frames)

        all_ious = []
        all_dices = []
        all_gt_coverages = []
        all_pred_precisions = []
        all_center_accuracies = []
        all_detection_successes = []

        for frame_id in sorted(common_frames, key=int):
            gt_frame = gt_data["labels"][frame_id]
            pred_frame = pred_data["labels"][frame_id]

            frame_result = self.analyze_frame(frame_id, gt_frame, pred_frame)

            if frame_result:
                video_results["frame_results"].append(frame_result)
                video_results["total_gt_objects"] += frame_result["gt_count"]
                video_results["total_pred_objects"] += frame_result["pred_count"]
                video_results["total_matches"] += frame_result["matches"]

                all_ious.extend(frame_result["iou_scores"])
                all_dices.extend(frame_result["dice_scores"])
                all_gt_coverages.extend(frame_result["gt_coverage_ratios"])
                all_pred_precisions.extend(frame_result["prediction_precisions"])
                all_center_accuracies.extend(frame_result["center_distance_accuracies"])
                all_detection_successes.extend(frame_result["detection_successes"])

        # Calculate averages
        video_results["avg_iou"] = np.mean(all_ious) if all_ious else 0.0
        video_results["avg_dice"] = np.mean(all_dices) if all_dices else 0.0
        video_results["avg_gt_coverage"] = (
            np.mean(all_gt_coverages) if all_gt_coverages else 0.0
        )
        video_results["avg_prediction_precision"] = (
            np.mean(all_pred_precisions) if all_pred_precisions else 0.0
        )
        video_results["avg_center_accuracy"] = (
            np.mean(all_center_accuracies) if all_center_accuracies else 0.0
        )
        video_results["detection_success_rate"] = (
            np.mean(all_detection_successes) if all_detection_successes else 0.0
        )
        video_results["total_frames_analyzed"] = len(video_results["frame_results"])

        return video_results

    def run_analysis(self):
        """Run the complete analysis"""
        print("Starting enhanced TTI validation analysis...")

        # Create file mapping
        self.create_file_mapping()

        # Analyze each video pair
        for uuid, file_info in self.file_mapping.items():
            try:
                video_result = self.analyze_video_pair(uuid, file_info)
                self.results.append(video_result)
            except Exception as e:
                print(f"Error analyzing {file_info['video_title']}: {e}")

        return self.results

    def generate_structured_output(self):
        """Generate structured output with enhanced metrics"""
        if not self.results:
            print("No results to summarize")
            return None

        # Prepare structured output
        structured_output = {
            "iou": {},
            "dice": {},
            "gt_coverage": {},
            "prediction_precision": {},
            "center_accuracy": {},
            "detection_success_rate": {},
            "statistical_summary": {},
            "visualization_files": [],
        }

        print("\n" + "=" * 80)
        print("ENHANCED TTI VALIDATION RESULTS - PROJECT I")
        print("=" * 80)

        # Collect all scores
        total_iou_scores = []
        total_dice_scores = []
        total_gt_coverage_scores = []
        total_pred_precision_scores = []
        total_center_accuracy_scores = []
        total_detection_success_rates = []

        # Individual video results
        print("\nDETAILED RESULTS BY VIDEO:")
        print("-" * 60)
        print(
            f"{'Video':<35} {'IoU':<6} {'Dice':<6} {'GT_Cov':<7} {'Pred_Prec':<9} {'Center':<7} {'Det_Rate':<8}"
        )
        print("-" * 60)

        for result in self.results:
            video_name = result["video_title"]
            iou_score = result["avg_iou"]
            dice_score = result["avg_dice"]
            gt_coverage = result["avg_gt_coverage"]
            pred_precision = result["avg_prediction_precision"]
            center_accuracy = result["avg_center_accuracy"]
            detection_rate = result["detection_success_rate"]

            # Store in structured output
            structured_output["iou"][video_name] = iou_score
            structured_output["dice"][video_name] = dice_score
            structured_output["gt_coverage"][video_name] = gt_coverage
            structured_output["prediction_precision"][video_name] = pred_precision
            structured_output["center_accuracy"][video_name] = center_accuracy
            structured_output["detection_success_rate"][video_name] = detection_rate

            # Add to totals
            total_iou_scores.append(iou_score)
            total_dice_scores.append(dice_score)
            total_gt_coverage_scores.append(gt_coverage)
            total_pred_precision_scores.append(pred_precision)
            total_center_accuracy_scores.append(center_accuracy)
            total_detection_success_rates.append(detection_rate)

            # Truncate video name for display
            display_name = (
                video_name[:32] + "..." if len(video_name) > 32 else video_name
            )
            print(
                f"{display_name:<35} {iou_score:.3f}  {dice_score:.3f}  {gt_coverage:.3f}   {pred_precision:.3f}     {center_accuracy:.3f}   {detection_rate:.3f}"
            )

        # Overall averages
        overall_iou = np.mean(total_iou_scores) if total_iou_scores else 0.0
        overall_dice = np.mean(total_dice_scores) if total_dice_scores else 0.0
        overall_gt_coverage = (
            np.mean(total_gt_coverage_scores) if total_gt_coverage_scores else 0.0
        )
        overall_pred_precision = (
            np.mean(total_pred_precision_scores) if total_pred_precision_scores else 0.0
        )
        overall_center_accuracy = (
            np.mean(total_center_accuracy_scores)
            if total_center_accuracy_scores
            else 0.0
        )
        overall_detection_rate = (
            np.mean(total_detection_success_rates)
            if total_detection_success_rates
            else 0.0
        )

        print("-" * 60)
        print(
            f"{'OVERALL':<35} {overall_iou:.3f}  {overall_dice:.3f}  {overall_gt_coverage:.3f}   {overall_pred_precision:.3f}     {overall_center_accuracy:.3f}   {overall_detection_rate:.3f}"
        )

        # Store overall results
        structured_output["iou"]["overall"] = overall_iou
        structured_output["dice"]["overall"] = overall_dice
        structured_output["gt_coverage"]["overall"] = overall_gt_coverage
        structured_output["prediction_precision"]["overall"] = overall_pred_precision
        structured_output["center_accuracy"]["overall"] = overall_center_accuracy
        structured_output["detection_success_rate"]["overall"] = overall_detection_rate

        # Statistical Summary
        safe_results = [r for r in self.results if r["video_type"] == "safe"]
        bdi_results = [r for r in self.results if r["video_type"] == "bdi"]

        safe_gt_coverages = [r["avg_gt_coverage"] for r in safe_results]
        bdi_gt_coverages = [r["avg_gt_coverage"] for r in bdi_results]

        structured_output["statistical_summary"] = {
            "total_videos": len(self.results),
            "safe_videos": len(safe_results),
            "bdi_videos": len(bdi_results),
            "gt_coverage_statistics": {
                "overall_mean": float(overall_gt_coverage),
                "overall_std": float(np.std(total_gt_coverage_scores)),
                "safe_mean": (
                    float(np.mean(safe_gt_coverages)) if safe_gt_coverages else 0.0
                ),
                "safe_std": (
                    float(np.std(safe_gt_coverages)) if safe_gt_coverages else 0.0
                ),
                "bdi_mean": (
                    float(np.mean(bdi_gt_coverages)) if bdi_gt_coverages else 0.0
                ),
                "bdi_std": float(np.std(bdi_gt_coverages)) if bdi_gt_coverages else 0.0,
            },
            "iou_statistics": {
                "overall_mean": float(overall_iou),
                "overall_std": float(np.std(total_iou_scores)),
                "safe_mean": (
                    float(np.mean([r["avg_iou"] for r in safe_results]))
                    if safe_results
                    else 0.0
                ),
                "safe_std": (
                    float(np.std([r["avg_iou"] for r in safe_results]))
                    if safe_results
                    else 0.0
                ),
                "bdi_mean": (
                    float(np.mean([r["avg_iou"] for r in bdi_results]))
                    if bdi_results
                    else 0.0
                ),
                "bdi_std": (
                    float(np.std([r["avg_iou"] for r in bdi_results]))
                    if bdi_results
                    else 0.0
                ),
            },
            "dice_statistics": {
                "overall_mean": float(overall_dice),
                "overall_std": float(np.std(total_dice_scores)),
                "safe_mean": (
                    float(np.mean([r["avg_dice"] for r in safe_results]))
                    if safe_results
                    else 0.0
                ),
                "safe_std": (
                    float(np.std([r["avg_dice"] for r in safe_results]))
                    if safe_results
                    else 0.0
                ),
                "bdi_mean": (
                    float(np.mean([r["avg_dice"] for r in bdi_results]))
                    if bdi_results
                    else 0.0
                ),
                "bdi_std": (
                    float(np.std([r["avg_dice"] for r in bdi_results]))
                    if bdi_results
                    else 0.0
                ),
            },
            "detection_success_statistics": {
                "overall_mean": float(overall_detection_rate),
                "overall_std": float(np.std(total_detection_success_rates)),
                "safe_mean": (
                    float(np.mean([r["detection_success_rate"] for r in safe_results]))
                    if safe_results
                    else 0.0
                ),
                "safe_std": (
                    float(np.std([r["detection_success_rate"] for r in safe_results]))
                    if safe_results
                    else 0.0
                ),
                "bdi_mean": (
                    float(np.mean([r["detection_success_rate"] for r in bdi_results]))
                    if bdi_results
                    else 0.0
                ),
                "bdi_std": (
                    float(np.std([r["detection_success_rate"] for r in bdi_results]))
                    if bdi_results
                    else 0.0
                ),
            },
        }

        print("\nSTATISTICAL SUMMARY:")
        print("-" * 50)
        stats = structured_output["statistical_summary"]
        print(f"Total videos analyzed: {stats['total_videos']}")
        print(f"Safe videos: {stats['safe_videos']}")
        print(f"BDI videos: {stats['bdi_videos']}")

        print(f"\nGT Coverage Statistics:")
        print(
            f"  Overall: {stats['gt_coverage_statistics']['overall_mean']:.3f} ± {stats['gt_coverage_statistics']['overall_std']:.3f}"
        )
        print(
            f"  Safe:    {stats['gt_coverage_statistics']['safe_mean']:.3f} ± {stats['gt_coverage_statistics']['safe_std']:.3f}"
        )
        print(
            f"  BDI:     {stats['gt_coverage_statistics']['bdi_mean']:.3f} ± {stats['gt_coverage_statistics']['bdi_std']:.3f}"
        )

        print(f"\nDetection Success Rate:")
        print(
            f"  Overall: {stats['detection_success_statistics']['overall_mean']:.3f} ± {stats['detection_success_statistics']['overall_std']:.3f}"
        )
        print(
            f"  Safe:    {stats['detection_success_statistics']['safe_mean']:.3f} ± {stats['detection_success_statistics']['safe_std']:.3f}"
        )
        print(
            f"  BDI:     {stats['detection_success_statistics']['bdi_mean']:.3f} ± {stats['detection_success_statistics']['bdi_std']:.3f}"
        )

        print(f"\nTraditional IoU Statistics:")
        print(
            f"  Overall: {stats['iou_statistics']['overall_mean']:.3f} ± {stats['iou_statistics']['overall_std']:.3f}"
        )
        print(
            f"  Safe:    {stats['iou_statistics']['safe_mean']:.3f} ± {stats['iou_statistics']['safe_std']:.3f}"
        )
        print(
            f"  BDI:     {stats['iou_statistics']['bdi_mean']:.3f} ± {stats['iou_statistics']['bdi_std']:.3f}"
        )

        return structured_output

    def create_enhanced_visualizations(self, structured_output):
        """Create enhanced visualization plots"""
        if not self.results:
            return []

        visualization_files = []

        # Prepare data
        safe_results = [r for r in self.results if r["video_type"] == "safe"]
        bdi_results = [r for r in self.results if r["video_type"] == "bdi"]

        # Extract metrics for each group
        safe_metrics = {
            "iou": [r["avg_iou"] for r in safe_results],
            "dice": [r["avg_dice"] for r in safe_results],
            "gt_coverage": [r["avg_gt_coverage"] for r in safe_results],
            "detection_rate": [r["detection_success_rate"] for r in safe_results],
        }

        bdi_metrics = {
            "iou": [r["avg_iou"] for r in bdi_results],
            "dice": [r["avg_dice"] for r in bdi_results],
            "gt_coverage": [r["avg_gt_coverage"] for r in bdi_results],
            "detection_rate": [r["detection_success_rate"] for r in bdi_results],
        }

        # Create comprehensive visualization
        fig = plt.figure(figsize=(20, 16))

        # Create a 3x3 grid for better layout
        gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)

        # 1. GT Coverage comparison bar chart (top-left)
        ax1 = fig.add_subplot(gs[0, 0])
        if safe_metrics["gt_coverage"] and bdi_metrics["gt_coverage"]:
            means = [
                np.mean(safe_metrics["gt_coverage"]),
                np.mean(bdi_metrics["gt_coverage"]),
            ]
            stds = [
                np.std(safe_metrics["gt_coverage"]),
                np.std(bdi_metrics["gt_coverage"]),
            ]
            bars = ax1.bar(
                ["Safe", "BDI"],
                means,
                yerr=stds,
                capsize=5,
                color=["lightblue", "lightcoral"],
                alpha=0.7,
                edgecolor="black",
            )

            # Add value labels
            for i, (bar, mean_val) in enumerate(zip(bars, means)):
                ax1.text(
                    bar.get_x() + bar.get_width() / 2,
                    mean_val + stds[i] + 0.02,
                    f"{mean_val:.3f}",
                    ha="center",
                    va="bottom",
                    fontweight="bold",
                )

        ax1.set_title("GT Coverage Ratio by Video Type", fontsize=14, fontweight="bold")
        ax1.set_ylabel("GT Coverage Ratio", fontsize=12)
        ax1.set_ylim(0, 1)
        ax1.grid(True, alpha=0.3)

        # 2. Detection Success Rate comparison (top-center)
        ax2 = fig.add_subplot(gs[0, 1])
        if safe_metrics["detection_rate"] and bdi_metrics["detection_rate"]:
            means = [
                np.mean(safe_metrics["detection_rate"]),
                np.mean(bdi_metrics["detection_rate"]),
            ]
            stds = [
                np.std(safe_metrics["detection_rate"]),
                np.std(bdi_metrics["detection_rate"]),
            ]
            bars = ax2.bar(
                ["Safe", "BDI"],
                means,
                yerr=stds,
                capsize=5,
                color=["lightgreen", "lightsalmon"],
                alpha=0.7,
                edgecolor="black",
            )

            # Add value labels
            for i, (bar, mean_val) in enumerate(zip(bars, means)):
                ax2.text(
                    bar.get_x() + bar.get_width() / 2,
                    mean_val + stds[i] + 0.02,
                    f"{mean_val:.3f}",
                    ha="center",
                    va="bottom",
                    fontweight="bold",
                )

        ax2.set_title(
            "Detection Success Rate by Video Type", fontsize=14, fontweight="bold"
        )
        ax2.set_ylabel("Detection Success Rate", fontsize=12)
        ax2.set_ylim(0, 1)
        ax2.grid(True, alpha=0.3)

        # 3. Traditional IoU comparison (top-right)
        ax3 = fig.add_subplot(gs[0, 2])
        if safe_metrics["iou"] and bdi_metrics["iou"]:
            means = [np.mean(safe_metrics["iou"]), np.mean(bdi_metrics["iou"])]
            stds = [np.std(safe_metrics["iou"]), np.std(bdi_metrics["iou"])]
            bars = ax3.bar(
                ["Safe", "BDI"],
                means,
                yerr=stds,
                capsize=5,
                color=["gold", "orange"],
                alpha=0.7,
                edgecolor="black",
            )

            # Add value labels
            for i, (bar, mean_val) in enumerate(zip(bars, means)):
                ax3.text(
                    bar.get_x() + bar.get_width() / 2,
                    mean_val + stds[i] + 0.02,
                    f"{mean_val:.3f}",
                    ha="center",
                    va="bottom",
                    fontweight="bold",
                )

        ax3.set_title("Traditional IoU by Video Type", fontsize=14, fontweight="bold")
        ax3.set_ylabel("IoU Score", fontsize=12)
        ax3.set_ylim(0, 1)
        ax3.grid(True, alpha=0.3)

        # 4. GT Coverage distribution box plot (middle-left)
        ax4 = fig.add_subplot(gs[1, 0])
        all_gt_coverage = safe_metrics["gt_coverage"] + bdi_metrics["gt_coverage"]
        gt_coverage_types = ["Safe"] * len(safe_metrics["gt_coverage"]) + ["BDI"] * len(
            bdi_metrics["gt_coverage"]
        )

        if all_gt_coverage:
            data_gt_coverage = pd.DataFrame(
                {"GT_Coverage": all_gt_coverage, "Type": gt_coverage_types}
            )
            sns.boxplot(
                data=data_gt_coverage,
                x="Type",
                y="GT_Coverage",
                ax=ax4,
                palette=["lightblue", "lightcoral"],
            )
            ax4.set_title("GT Coverage Distribution", fontsize=14, fontweight="bold")
            ax4.set_ylabel("GT Coverage Ratio", fontsize=12)
            ax4.grid(True, alpha=0.3)

        # 5. Detection Success Rate distribution (middle-center)
        ax5 = fig.add_subplot(gs[1, 1])
        all_detection_rates = (
            safe_metrics["detection_rate"] + bdi_metrics["detection_rate"]
        )
        detection_rate_types = ["Safe"] * len(safe_metrics["detection_rate"]) + [
            "BDI"
        ] * len(bdi_metrics["detection_rate"])

        if all_detection_rates:
            data_detection = pd.DataFrame(
                {"Detection_Rate": all_detection_rates, "Type": detection_rate_types}
            )
            sns.boxplot(
                data=data_detection,
                x="Type",
                y="Detection_Rate",
                ax=ax5,
                palette=["lightgreen", "lightsalmon"],
            )
            ax5.set_title(
                "Detection Success Rate Distribution", fontsize=14, fontweight="bold"
            )
            ax5.set_ylabel("Detection Success Rate", fontsize=12)
            ax5.grid(True, alpha=0.3)

        # 6. Scatter plot: GT Coverage vs IoU (middle-right)
        ax6 = fig.add_subplot(gs[1, 2])
        all_iou = [r["avg_iou"] for r in self.results]
        all_gt_cov = [r["avg_gt_coverage"] for r in self.results]
        colors = ["blue" if r["video_type"] == "safe" else "red" for r in self.results]

        scatter = ax6.scatter(all_iou, all_gt_cov, c=colors, alpha=0.7, s=60)
        ax6.set_xlabel("IoU Score", fontsize=12)
        ax6.set_ylabel("GT Coverage Ratio", fontsize=12)
        ax6.set_title("GT Coverage vs IoU Correlation", fontsize=14, fontweight="bold")
        ax6.grid(True, alpha=0.3)

        # Add trend line
        if len(all_iou) > 1:
            z = np.polyfit(all_iou, all_gt_cov, 1)
            p = np.poly1d(z)
            ax6.plot(sorted(all_iou), p(sorted(all_iou)), "k--", alpha=0.8, linewidth=2)

            # Calculate correlation
            correlation = np.corrcoef(all_iou, all_gt_cov)[0, 1]
            ax6.text(
                0.05,
                0.95,
                f"Correlation: {correlation:.3f}",
                transform=ax6.transAxes,
                bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
            )

        # 7. Individual video GT Coverage scores (bottom row, spanning 2 columns)
        ax7 = fig.add_subplot(gs[2, :2])
        video_names = [r["video_title"] for r in self.results]
        gt_coverage_scores = [r["avg_gt_coverage"] for r in self.results]
        video_colors = [
            "blue" if r["video_type"] == "safe" else "red" for r in self.results
        ]

        bars = ax7.bar(
            range(len(video_names)), gt_coverage_scores, color=video_colors, alpha=0.7
        )
        ax7.set_title(
            "GT Coverage Ratio by Individual Video", fontsize=14, fontweight="bold"
        )
        ax7.set_ylabel("GT Coverage Ratio", fontsize=12)
        ax7.set_xticks(range(len(video_names)))
        ax7.set_xticklabels(
            [name[:15] + "..." if len(name) > 15 else name for name in video_names],
            rotation=45,
            ha="right",
        )
        ax7.grid(True, alpha=0.3)

        # Add value labels on bars
        for i, bar in enumerate(bars):
            height = bar.get_height()
            ax7.text(
                bar.get_x() + bar.get_width() / 2.0,
                height + 0.01,
                f"{height:.3f}",
                ha="center",
                va="bottom",
                fontsize=8,
            )

        # 8. Metrics comparison radar-like chart (bottom-right)
        ax8 = fig.add_subplot(gs[2, 2])
        metrics_names = ["GT Coverage", "Detection Rate", "IoU", "Dice"]
        safe_means = [
            np.mean(safe_metrics["gt_coverage"]) if safe_metrics["gt_coverage"] else 0,
            (
                np.mean(safe_metrics["detection_rate"])
                if safe_metrics["detection_rate"]
                else 0
            ),
            np.mean(safe_metrics["iou"]) if safe_metrics["iou"] else 0,
            np.mean(safe_metrics["dice"]) if safe_metrics["dice"] else 0,
        ]
        bdi_means = [
            np.mean(bdi_metrics["gt_coverage"]) if bdi_metrics["gt_coverage"] else 0,
            (
                np.mean(bdi_metrics["detection_rate"])
                if bdi_metrics["detection_rate"]
                else 0
            ),
            np.mean(bdi_metrics["iou"]) if bdi_metrics["iou"] else 0,
            np.mean(bdi_metrics["dice"]) if bdi_metrics["dice"] else 0,
        ]

        x_pos = np.arange(len(metrics_names))
        width = 0.35

        bars1 = ax8.bar(
            x_pos - width / 2,
            safe_means,
            width,
            label="Safe",
            color="lightblue",
            alpha=0.7,
        )
        bars2 = ax8.bar(
            x_pos + width / 2,
            bdi_means,
            width,
            label="BDI",
            color="lightcoral",
            alpha=0.7,
        )

        ax8.set_xlabel("Metrics", fontsize=12)
        ax8.set_ylabel("Score", fontsize=12)
        ax8.set_title("All Metrics Comparison", fontsize=14, fontweight="bold")
        ax8.set_xticks(x_pos)
        ax8.set_xticklabels(metrics_names, rotation=45, ha="right")
        ax8.legend()
        ax8.grid(True, alpha=0.3)
        ax8.set_ylim(0, 1)

        # Add value labels on bars
        for bars in [bars1, bars2]:
            for bar in bars:
                height = bar.get_height()
                ax8.text(
                    bar.get_x() + bar.get_width() / 2.0,
                    height + 0.01,
                    f"{height:.3f}",
                    ha="center",
                    va="bottom",
                    fontsize=8,
                )

        # Add legend for scatter plot
        from matplotlib.patches import Patch

        legend_elements = [
            Patch(facecolor="blue", alpha=0.7, label="Safe"),
            Patch(facecolor="red", alpha=0.7, label="BDI"),
        ]
        ax6.legend(handles=legend_elements, loc="upper left")

        plt.suptitle(
            "Enhanced TTI Validation Analysis with GT Coverage Metrics",
            fontsize=16,
            fontweight="bold",
            y=0.98,
        )

        # Save enhanced visualization
        enhanced_viz_file = f"{self.output_prefix}_enhanced_validation_plots.png"
        plt.savefig(enhanced_viz_file, dpi=300, bbox_inches="tight", facecolor="white")
        visualization_files.append(enhanced_viz_file)
        plt.show()

        # Create individual metrics comparison plot
        fig2, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig2.suptitle(
            "Detailed Metric Comparisons: Safe vs BDI Videos",
            fontsize=16,
            fontweight="bold",
        )

        metrics_data = [
            (
                "GT Coverage Ratio",
                safe_metrics["gt_coverage"],
                bdi_metrics["gt_coverage"],
                "Blues",
                "Reds",
            ),
            (
                "Detection Success Rate",
                safe_metrics["detection_rate"],
                bdi_metrics["detection_rate"],
                "Greens",
                "Oranges",
            ),
            ("IoU Score", safe_metrics["iou"], bdi_metrics["iou"], "Purples", "YlOrRd"),
            (
                "Dice Score",
                safe_metrics["dice"],
                bdi_metrics["dice"],
                "viridis",
                "plasma",
            ),
        ]

        for idx, (metric_name, safe_data, bdi_data, safe_cmap, bdi_cmap) in enumerate(
            metrics_data
        ):
            ax = axes[idx // 2, idx % 2]

            if safe_data and bdi_data:
                # Create histogram comparison
                bins = np.linspace(0, 1, 21)
                ax.hist(
                    safe_data,
                    bins=bins,
                    alpha=0.6,
                    label="Safe",
                    color="lightblue",
                    edgecolor="black",
                )
                ax.hist(
                    bdi_data,
                    bins=bins,
                    alpha=0.6,
                    label="BDI",
                    color="lightcoral",
                    edgecolor="black",
                )

                ax.set_xlabel(metric_name, fontsize=12)
                ax.set_ylabel("Frequency", fontsize=12)
                ax.set_title(
                    f"{metric_name} Distribution", fontsize=14, fontweight="bold"
                )
                ax.legend()
                ax.grid(True, alpha=0.3)

                # Add statistics text
                safe_mean = np.mean(safe_data)
                bdi_mean = np.mean(bdi_data)
                safe_std = np.std(safe_data)
                bdi_std = np.std(bdi_data)

                stats_text = f"Safe: {safe_mean:.3f} ± {safe_std:.3f}\nBDI: {bdi_mean:.3f} ± {bdi_std:.3f}"
                ax.text(
                    0.02,
                    0.98,
                    stats_text,
                    transform=ax.transAxes,
                    verticalalignment="top",
                    bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
                )

        plt.tight_layout()

        detailed_viz_file = f"{self.output_prefix}_detailed_metrics_comparison.png"
        plt.savefig(detailed_viz_file, dpi=300, bbox_inches="tight", facecolor="white")
        visualization_files.append(detailed_viz_file)
        plt.show()

        structured_output["visualization_files"] = visualization_files

        print(f"\nENHANCED VISUALIZATIONS:")
        print("-" * 50)
        for viz_file in visualization_files:
            print(f"  Created: {viz_file}")

        return visualization_files


def main():
    """Main execution function"""
    parser = argparse.ArgumentParser(
        description="Enhanced TTI Model Validation Analysis with GT Coverage - Project I",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python enhanced_project_i_analysis.py --tti-inference-dir lorenz-results --gt-dir ground_truth_annos --videos-dir videos
  python enhanced_project_i_analysis.py -i ./lorenz-results -g ./ground_truth_annos -v ./videos --output-prefix enhanced_tti_
        """,
    )

    parser.add_argument(
        "--tti-inference-dir",
        "-i",
        required=True,
        help="Path to TTI inference output directory (contains JSON files with _vit_eval.json suffix)",
    )

    parser.add_argument(
        "--gt-dir",
        "-g",
        required=True,
        help="Path to ground truth annotations directory (contains JSON files)",
    )

    parser.add_argument(
        "--videos-dir", "-v", required=True, help="Path to videos directory"
    )

    parser.add_argument(
        "--output-prefix",
        "-o",
        default="enhanced_project_i_",
        help="Prefix for output files (default: 'enhanced_project_i_')",
    )

    args = parser.parse_args()

    # Validate paths exist
    for path_name, path_value in [
        ("TTI output directory", args.tti_inference_dir),
        ("Ground truth directory", args.gt_dir),
        ("Videos directory", args.videos_dir),
    ]:
        if not os.path.exists(path_value):
            print(f"Error: {path_name} does not exist: {path_value}")
            print(f"Please check the path and try again.")
            return None

    print(f"TTI output directory: {args.tti_inference_dir}")
    print(f"Ground truth directory: {args.gt_dir}")
    print(f"Videos directory: {args.videos_dir}")
    print(f"Output prefix: {args.output_prefix}")
    print()

    # Initialize enhanced analyzer
    analyzer = EnhancedTTIValidationAnalyzer(
        args.tti_inference_dir, args.gt_dir, args.videos_dir, args.output_prefix
    )

    # Run analysis
    results = analyzer.run_analysis()

    # Generate structured output
    structured_output = analyzer.generate_structured_output()

    # Create enhanced visualizations
    if structured_output:
        analyzer.create_enhanced_visualizations(structured_output)

    # Save all outputs
    output_files = {
        "structured_results": f"{args.output_prefix}_structured_results.json",
        "detailed_results": f"{args.output_prefix}_detailed_results.json",
        "summary_csv": f"{args.output_prefix}_summary.csv",
        "enhanced_csv": f"{args.output_prefix}_enhanced_metrics.csv",
    }

    if structured_output:
        # Save structured output
        with open(output_files["structured_results"], "w") as f:
            json.dump(structured_output, f, indent=2)

        # Save detailed results
        with open(output_files["detailed_results"], "w") as f:
            json.dump(results, f, indent=2)

        # Create and save enhanced summary CSV
        summary_data = []
        for result in results:
            summary_data.append(
                {
                    "Video": result["video_title"],
                    "Type": result["video_type"],
                    "Frames_Analyzed": result["total_frames_analyzed"],
                    "GT_Objects": result["total_gt_objects"],
                    "Pred_Objects": result["total_pred_objects"],
                    "Matches": result["total_matches"],
                    "IoU_Score": result["avg_iou"],
                    "Dice_Score": result["avg_dice"],
                    "GT_Coverage": result["avg_gt_coverage"],
                    "Prediction_Precision": result["avg_prediction_precision"],
                    "Center_Accuracy": result["avg_center_accuracy"],
                    "Detection_Success_Rate": result["detection_success_rate"],
                }
            )

        df = pd.DataFrame(summary_data)
        df.to_csv(output_files["summary_csv"], index=False)

        # Create enhanced metrics CSV for deeper analysis
        enhanced_data = []
        for result in results:
            for frame_result in result["frame_results"]:
                frame_id = frame_result["frame_id"]
                for i in range(len(frame_result["iou_scores"])):
                    enhanced_data.append(
                        {
                            "Video": result["video_title"],
                            "Video_Type": result["video_type"],
                            "Frame_ID": frame_id,
                            "Match_Index": i,
                            "IoU": frame_result["iou_scores"][i],
                            "Dice": frame_result["dice_scores"][i],
                            "GT_Coverage": frame_result["gt_coverage_ratios"][i],
                            "Pred_Precision": frame_result["prediction_precisions"][i],
                            "Center_Accuracy": frame_result[
                                "center_distance_accuracies"
                            ][i],
                            "Detection_Success": frame_result["detection_successes"][i],
                        }
                    )

        if enhanced_data:
            enhanced_df = pd.DataFrame(enhanced_data)
            enhanced_df.to_csv(output_files["enhanced_csv"], index=False)

        print(f"\nOUTPUT FILES CREATED:")
        print("-" * 50)
        for desc, filename in output_files.items():
            if os.path.exists(filename):
                print(f"  {desc:<25}: {filename}")

        print(f"\nENHANCED PROJECT I ANALYSIS COMPLETE!")
        print("=" * 60)
        print("Key Deliverables:")
        print("✓ Traditional IoU & Dice scores")
        print("✓ GT Coverage Ratio (practical TTI detection metric)")
        print("✓ Detection Success Rate (clinical relevance)")
        print("✓ Prediction Precision (false positive analysis)")
        print("✓ Center Distance Accuracy (localization quality)")
        print("✓ Enhanced visualizations comparing Safe vs BDI")
        print("✓ Statistical summaries for all metrics")
        print("✓ Frame-level detailed analysis")

        print(f"\nKEY INSIGHTS:")
        if structured_output:
            stats = structured_output["statistical_summary"]
            print(
                f"📊 Overall GT Coverage: {stats['gt_coverage_statistics']['overall_mean']:.1%}"
            )
            print(
                f"📊 Overall Detection Success: {stats['detection_success_statistics']['overall_mean']:.1%}"
            )
            print(f"📊 Traditional IoU: {stats['iou_statistics']['overall_mean']:.3f}")

            if stats["safe_videos"] > 0 and stats["bdi_videos"] > 0:
                safe_gt_cov = stats["gt_coverage_statistics"]["safe_mean"]
                bdi_gt_cov = stats["gt_coverage_statistics"]["bdi_mean"]
                print(
                    f"📊 Safe vs BDI GT Coverage: {safe_gt_cov:.3f} vs {bdi_gt_cov:.3f}"
                )

    return structured_output


if __name__ == "__main__":
    main()
