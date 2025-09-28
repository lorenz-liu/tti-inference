#!/usr/bin/env python3
"""
Project I: IoU/Dice Score Calculations for TTI Model Validation
Compares TTI model predictions (segmentations) with ground truth bounding box annotations
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


class TTIValidationAnalyzer:
    def __init__(
        self,
        tti_output_dir: str,
        ground_truth_dir: str,
        videos_dir: str,
        output_prefix: str = "project_i",
    ):
        """
        Initialize the TTI validation analyzer

        Args:
            tti_output_dir: Path to inference-by-lorenz directory
            ground_truth_dir: Path to ground truth annotations directory
            videos_dir: Path to videos directory
            output_prefix: Prefix for output files (default: "project_i")
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

    def bbox_to_mask(self, bbox: Dict, width: int, height: int) -> np.ndarray:
        """Convert bounding box to binary mask"""
        mask = np.zeros((height, width), dtype=np.uint8)

        # Convert normalized coordinates to pixel coordinates
        x = int(bbox["x"] * width)
        y = int(bbox["y"] * height)
        w = int(bbox["w"] * width)
        h = int(bbox["h"] * height)

        # Ensure coordinates are within bounds
        x = max(0, min(x, width - 1))
        y = max(0, min(y, height - 1))
        w = max(1, min(w, width - x))
        h = max(1, min(h, height - y))

        mask[y : y + h, x : x + w] = 1
        return mask

    def calculate_iou(self, bbox1: Dict, bbox2: Dict, width: int, height: int) -> float:
        """Calculate IoU between two bounding boxes"""
        # Convert to pixel coordinates
        x1, y1, w1, h1 = (
            bbox1["x"] * width,
            bbox1["y"] * height,
            bbox1["w"] * width,
            bbox1["h"] * height,
        )
        x2, y2, w2, h2 = (
            bbox2["x"] * width,
            bbox2["y"] * height,
            bbox2["w"] * width,
            bbox2["h"] * height,
        )

        # Calculate intersection
        x_left = max(x1, x2)
        y_top = max(y1, y2)
        x_right = min(x1 + w1, x2 + w2)
        y_bottom = min(y1 + h1, y2 + h2)

        if x_right < x_left or y_bottom < y_top:
            return 0.0

        intersection = (x_right - x_left) * (y_bottom - y_top)

        # Calculate union
        area1 = w1 * h1
        area2 = w2 * h2
        union = area1 + area2 - intersection

        return intersection / union if union > 0 else 0.0

    def calculate_dice(self, mask1: np.ndarray, mask2: np.ndarray) -> float:
        """Calculate Dice coefficient between two binary masks"""
        intersection = np.sum(mask1 * mask2)
        total = np.sum(mask1) + np.sum(mask2)

        return (2.0 * intersection) / total if total > 0 else 0.0

    def find_best_match(
        self, gt_objects: List, pred_objects: List, width: int, height: int
    ) -> List[Tuple]:
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

                iou = self.calculate_iou(
                    gt_obj["boundingBox"], pred_obj["boundingBox"], width, height
                )
                if iou > best_iou:
                    best_iou = iou
                    best_match = pred_obj

            if best_match is not None:
                matches.append((gt_obj, best_match, best_iou))

        return matches

    def analyze_frame(
        self,
        frame_id: str,
        gt_frame_data: Dict,
        pred_frame_data: Dict,
        width: int,
        height: int,
    ) -> Dict:
        """Analyze a single frame"""
        gt_objects = gt_frame_data.get("objects", [])
        pred_objects = pred_frame_data.get("objects", [])

        if not gt_objects and not pred_objects:
            return None

        # Find best matches
        matches = self.find_best_match(gt_objects, pred_objects, width, height)

        frame_results = {
            "frame_id": frame_id,
            "gt_count": len(gt_objects),
            "pred_count": len(pred_objects),
            "matches": len(matches),
            "iou_scores": [],
            "dice_scores": [],
        }

        for gt_obj, pred_obj, iou in matches:
            # Calculate Dice score
            gt_mask = self.bbox_to_mask(gt_obj["boundingBox"], width, height)
            pred_mask = self.bbox_to_mask(pred_obj["boundingBox"], width, height)
            dice = self.calculate_dice(gt_mask, pred_mask)

            frame_results["iou_scores"].append(iou)
            frame_results["dice_scores"].append(dice)

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
        }

        # Get common frame IDs
        gt_frames = set(gt_data["labels"].keys())
        pred_frames = set(pred_data["labels"].keys())
        common_frames = gt_frames.intersection(pred_frames)

        all_ious = []
        all_dices = []

        for frame_id in sorted(common_frames, key=int):
            gt_frame = gt_data["labels"][frame_id]
            pred_frame = pred_data["labels"][frame_id]

            frame_result = self.analyze_frame(
                frame_id, gt_frame, pred_frame, width, height
            )

            if frame_result:
                video_results["frame_results"].append(frame_result)
                video_results["total_gt_objects"] += frame_result["gt_count"]
                video_results["total_pred_objects"] += frame_result["pred_count"]
                video_results["total_matches"] += frame_result["matches"]

                all_ious.extend(frame_result["iou_scores"])
                all_dices.extend(frame_result["dice_scores"])

        # Calculate averages
        video_results["avg_iou"] = np.mean(all_ious) if all_ious else 0.0
        video_results["avg_dice"] = np.mean(all_dices) if all_dices else 0.0
        video_results["total_frames_analyzed"] = len(video_results["frame_results"])

        return video_results

    def run_analysis(self):
        """Run the complete analysis"""
        print("Starting TTI validation analysis...")

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
        """Generate structured output with the requested format"""
        if not self.results:
            print("No results to summarize")
            return None

        # Prepare structured output
        structured_output = {
            "iou": {},
            "dice": {},
            "statistical_summary": {},
            "visualization_files": [],
        }

        print("\n" + "=" * 80)
        print("TTI VALIDATION RESULTS - PROJECT I")
        print("=" * 80)

        # IoU Results
        print("\nIoU SCORES:")
        print("-" * 40)

        total_iou_scores = []
        for result in self.results:
            video_name = result["video_title"]
            iou_score = result["avg_iou"]
            structured_output["iou"][video_name] = iou_score
            total_iou_scores.append(iou_score)
            print(f"  {video_name:<35} {iou_score:.3f}")

        overall_iou = np.mean(total_iou_scores) if total_iou_scores else 0.0
        structured_output["iou"]["overall"] = overall_iou
        print(f"  {'OVERALL':<35} {overall_iou:.3f}")

        # Dice Results
        print("\nDICE SCORES:")
        print("-" * 40)

        total_dice_scores = []
        for result in self.results:
            video_name = result["video_title"]
            dice_score = result["avg_dice"]
            structured_output["dice"][video_name] = dice_score
            total_dice_scores.append(dice_score)
            print(f"  {video_name:<35} {dice_score:.3f}")

        overall_dice = np.mean(total_dice_scores) if total_dice_scores else 0.0
        structured_output["dice"]["overall"] = overall_dice
        print(f"  {'OVERALL':<35} {overall_dice:.3f}")

        # Statistical Summary
        safe_results = [r for r in self.results if r["video_type"] == "safe"]
        bdi_results = [r for r in self.results if r["video_type"] == "bdi"]

        safe_ious = [r["avg_iou"] for r in safe_results]
        bdi_ious = [r["avg_iou"] for r in bdi_results]
        safe_dices = [r["avg_dice"] for r in safe_results]
        bdi_dices = [r["avg_dice"] for r in bdi_results]

        structured_output["statistical_summary"] = {
            "total_videos": len(self.results),
            "safe_videos": len(safe_results),
            "bdi_videos": len(bdi_results),
            "iou_statistics": {
                "overall_mean": float(np.mean(total_iou_scores)),
                "overall_std": float(np.std(total_iou_scores)),
                "safe_mean": float(np.mean(safe_ious)) if safe_ious else 0.0,
                "safe_std": float(np.std(safe_ious)) if safe_ious else 0.0,
                "bdi_mean": float(np.mean(bdi_ious)) if bdi_ious else 0.0,
                "bdi_std": float(np.std(bdi_ious)) if bdi_ious else 0.0,
            },
            "dice_statistics": {
                "overall_mean": float(np.mean(total_dice_scores)),
                "overall_std": float(np.std(total_dice_scores)),
                "safe_mean": float(np.mean(safe_dices)) if safe_dices else 0.0,
                "safe_std": float(np.std(safe_dices)) if safe_dices else 0.0,
                "bdi_mean": float(np.mean(bdi_dices)) if bdi_dices else 0.0,
                "bdi_std": float(np.std(bdi_dices)) if bdi_dices else 0.0,
            },
        }

        print("\nSTATISTICAL SUMMARY:")
        print("-" * 40)
        stats = structured_output["statistical_summary"]
        print(f"Total videos analyzed: {stats['total_videos']}")
        print(f"Safe videos: {stats['safe_videos']}")
        print(f"BDI videos: {stats['bdi_videos']}")
        print(f"\nIoU Statistics:")
        print(
            f"  Overall: {stats['iou_statistics']['overall_mean']:.3f} ± {stats['iou_statistics']['overall_std']:.3f}"
        )
        print(
            f"  Safe:    {stats['iou_statistics']['safe_mean']:.3f} ± {stats['iou_statistics']['safe_std']:.3f}"
        )
        print(
            f"  BDI:     {stats['iou_statistics']['bdi_mean']:.3f} ± {stats['iou_statistics']['bdi_std']:.3f}"
        )
        print(f"\nDice Statistics:")
        print(
            f"  Overall: {stats['dice_statistics']['overall_mean']:.3f} ± {stats['dice_statistics']['overall_std']:.3f}"
        )
        print(
            f"  Safe:    {stats['dice_statistics']['safe_mean']:.3f} ± {stats['dice_statistics']['safe_std']:.3f}"
        )
        print(
            f"  BDI:     {stats['dice_statistics']['bdi_mean']:.3f} ± {stats['dice_statistics']['bdi_std']:.3f}"
        )

        return structured_output

    def create_visualizations(self, structured_output):
        """Create visualization plots with structured output"""
        if not self.results:
            return []

        visualization_files = []

        # Prepare data for plotting
        safe_ious = []
        bdi_ious = []
        safe_dices = []
        bdi_dices = []
        video_names = []
        video_types = []

        for result in self.results:
            video_names.append(result["video_title"])
            video_types.append(result["video_type"])

            if result["video_type"] == "safe":
                safe_ious.append(result["avg_iou"])
                safe_dices.append(result["avg_dice"])
            else:
                bdi_ious.append(result["avg_iou"])
                bdi_dices.append(result["avg_dice"])

        # Create comprehensive visualization
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))

        # 1. IoU comparison bar chart
        if safe_ious and bdi_ious:
            ax1.bar(
                ["Safe", "BDI"],
                [np.mean(safe_ious), np.mean(bdi_ious)],
                yerr=[np.std(safe_ious), np.std(bdi_ious)],
                capsize=5,
                color=["lightblue", "lightcoral"],
                alpha=0.7,
                edgecolor="black",
            )
        ax1.set_title(
            "Average IoU Scores by Video Type", fontsize=14, fontweight="bold"
        )
        ax1.set_ylabel("IoU Score", fontsize=12)
        ax1.set_ylim(0, 1)
        ax1.grid(True, alpha=0.3)

        # Add value labels on bars
        if safe_ious and bdi_ious:
            ax1.text(
                0,
                np.mean(safe_ious) + 0.02,
                f"{np.mean(safe_ious):.3f}",
                ha="center",
                va="bottom",
                fontweight="bold",
            )
            ax1.text(
                1,
                np.mean(bdi_ious) + 0.02,
                f"{np.mean(bdi_ious):.3f}",
                ha="center",
                va="bottom",
                fontweight="bold",
            )

        # 2. Dice comparison bar chart
        if safe_dices and bdi_dices:
            ax2.bar(
                ["Safe", "BDI"],
                [np.mean(safe_dices), np.mean(bdi_dices)],
                yerr=[np.std(safe_dices), np.std(bdi_dices)],
                capsize=5,
                color=["lightgreen", "lightsalmon"],
                alpha=0.7,
                edgecolor="black",
            )
        ax2.set_title(
            "Average Dice Scores by Video Type", fontsize=14, fontweight="bold"
        )
        ax2.set_ylabel("Dice Score", fontsize=12)
        ax2.set_ylim(0, 1)
        ax2.grid(True, alpha=0.3)

        # Add value labels on bars
        if safe_dices and bdi_dices:
            ax2.text(
                0,
                np.mean(safe_dices) + 0.02,
                f"{np.mean(safe_dices):.3f}",
                ha="center",
                va="bottom",
                fontweight="bold",
            )
            ax2.text(
                1,
                np.mean(bdi_dices) + 0.02,
                f"{np.mean(bdi_dices):.3f}",
                ha="center",
                va="bottom",
                fontweight="bold",
            )

        # 3. IoU distribution box plot
        all_ious = safe_ious + bdi_ious
        iou_types = ["Safe"] * len(safe_ious) + ["BDI"] * len(bdi_ious)

        if all_ious:
            data_iou = pd.DataFrame({"IoU": all_ious, "Type": iou_types})
            sns.boxplot(
                data=data_iou,
                x="Type",
                y="IoU",
                ax=ax3,
                palette=["lightblue", "lightcoral"],
            )
            ax3.set_title("IoU Score Distribution", fontsize=14, fontweight="bold")
            ax3.set_ylabel("IoU Score", fontsize=12)
            ax3.grid(True, alpha=0.3)

        # 4. Dice distribution box plot
        all_dices = safe_dices + bdi_dices
        dice_types = ["Safe"] * len(safe_dices) + ["BDI"] * len(bdi_dices)

        if all_dices:
            data_dice = pd.DataFrame({"Dice": all_dices, "Type": dice_types})
            sns.boxplot(
                data=data_dice,
                x="Type",
                y="Dice",
                ax=ax4,
                palette=["lightgreen", "lightsalmon"],
            )
            ax4.set_title("Dice Score Distribution", fontsize=14, fontweight="bold")
            ax4.set_ylabel("Dice Score", fontsize=12)
            ax4.grid(True, alpha=0.3)

        plt.tight_layout()

        # Save main visualization
        main_viz_file = f"{self.output_prefix}_tti_validation_plots.png"
        plt.savefig(main_viz_file, dpi=300, bbox_inches="tight")
        visualization_files.append(main_viz_file)
        plt.show()

        # Create individual video scores plot
        if len(self.results) > 1:
            fig2, (ax5, ax6) = plt.subplots(2, 1, figsize=(14, 10))

            # Individual IoU scores
            colors = ["blue" if vt == "safe" else "red" for vt in video_types]
            all_ious_individual = [r["avg_iou"] for r in self.results]

            bars1 = ax5.bar(
                range(len(video_names)), all_ious_individual, color=colors, alpha=0.7
            )
            ax5.set_title(
                "IoU Scores by Individual Video", fontsize=14, fontweight="bold"
            )
            ax5.set_ylabel("IoU Score", fontsize=12)
            ax5.set_xticks(range(len(video_names)))
            ax5.set_xticklabels(
                [name[:20] + "..." if len(name) > 20 else name for name in video_names],
                rotation=45,
                ha="right",
            )
            ax5.grid(True, alpha=0.3)

            # Add value labels on bars
            for i, bar in enumerate(bars1):
                height = bar.get_height()
                ax5.text(
                    bar.get_x() + bar.get_width() / 2.0,
                    height + 0.01,
                    f"{height:.3f}",
                    ha="center",
                    va="bottom",
                    fontsize=8,
                )

            # Individual Dice scores
            all_dices_individual = [r["avg_dice"] for r in self.results]

            bars2 = ax6.bar(
                range(len(video_names)), all_dices_individual, color=colors, alpha=0.7
            )
            ax6.set_title(
                "Dice Scores by Individual Video", fontsize=14, fontweight="bold"
            )
            ax6.set_ylabel("Dice Score", fontsize=12)
            ax6.set_xticks(range(len(video_names)))
            ax6.set_xticklabels(
                [name[:20] + "..." if len(name) > 20 else name for name in video_names],
                rotation=45,
                ha="right",
            )
            ax6.grid(True, alpha=0.3)

            # Add value labels on bars
            for i, bar in enumerate(bars2):
                height = bar.get_height()
                ax6.text(
                    bar.get_x() + bar.get_width() / 2.0,
                    height + 0.01,
                    f"{height:.3f}",
                    ha="center",
                    va="bottom",
                    fontsize=8,
                )

            # Add legend
            from matplotlib.patches import Patch

            legend_elements = [
                Patch(facecolor="blue", alpha=0.7, label="Safe"),
                Patch(facecolor="red", alpha=0.7, label="BDI"),
            ]
            ax5.legend(handles=legend_elements, loc="upper right")
            ax6.legend(handles=legend_elements, loc="upper right")

            plt.tight_layout()

            individual_viz_file = f"{self.output_prefix}_individual_video_scores.png"
            plt.savefig(individual_viz_file, dpi=300, bbox_inches="tight")
            visualization_files.append(individual_viz_file)
            plt.show()

        structured_output["visualization_files"] = visualization_files

        print(f"\nVISUALIZATIONS:")
        print("-" * 40)
        for viz_file in visualization_files:
            print(f"  Created: {viz_file}")

        return visualization_files


def main():
    """Main execution function"""
    parser = argparse.ArgumentParser(
        description="TTI Model Validation Analysis - Project I",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python project_i_analysis.py --tti-inference-dir lorenz-results --gt-dir ground_truth_annos --videos-dir videos
  python project_i_analysis.py -i ./lorenz-results -g ./ground_truth_annos -v ./videos
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
        default="project_i_",
        help="Prefix for output files (default: 'project_i_')",
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

    # Initialize analyzer
    analyzer = TTIValidationAnalyzer(
        args.tti_inference_dir, args.gt_dir, args.videos_dir, args.output_prefix
    )

    # Run analysis
    results = analyzer.run_analysis()

    # Generate structured output
    structured_output = analyzer.generate_structured_output()

    # Create visualizations
    if structured_output:
        analyzer.create_visualizations(structured_output)

    # Save all outputs
    output_files = {
        "structured_results": f"{args.output_prefix}_structured_results.json",
        "detailed_results": f"{args.output_prefix}_detailed_results.json",
        "summary_csv": f"{args.output_prefix}_summary.csv",
    }

    if structured_output:
        # Save structured output
        with open(output_files["structured_results"], "w") as f:
            json.dump(structured_output, f, indent=2)

        # Save detailed results
        with open(output_files["detailed_results"], "w") as f:
            json.dump(results, f, indent=2)

        # Create and save summary CSV
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
                }
            )

        df = pd.DataFrame(summary_data)
        df.to_csv(output_files["summary_csv"], index=False)

        print(f"\nOUTPUT FILES CREATED:")
        print("-" * 40)
        for desc, filename in output_files.items():
            print(f"  {desc:<20}: {filename}")

        print(f"\nPROJECT I ANALYSIS COMPLETE!")
        print("=" * 50)
        print("Deliverables ready for submission:")
        print("✓ IoU scores per video and overall")
        print("✓ Dice scores per video and overall")
        print("✓ Statistical summary with means and standard deviations")
        print("✓ Visualizations comparing Safe vs BDI videos")
        print("✓ CSV summary for further analysis")

    return structured_output


if __name__ == "__main__":
    main()
