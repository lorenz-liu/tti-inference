#!/usr/bin/env python3
"""
ROC Analysis for TTI Detection Model
Analyzes YOLO TTI predictions against Ground Truth to generate ROC curves
"""

import argparse
import glob
import os
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (
    auc,
    confusion_matrix,
    roc_curve,
)


class TTIROCAnalyzer:
    def __init__(self, csv_dir: str, output_prefix: str = "roc_analysis"):
        """
        Initialize the TTI ROC analyzer

        Args:
            csv_dir: Path to directory containing CSV files with TTI annotations
            output_prefix: Prefix for output files (default: "roc_analysis")
        """
        self.csv_dir = Path(csv_dir)
        self.output_prefix = output_prefix
        self.results = []

    def load_csv_data(self, csv_file: str) -> pd.DataFrame:
        """Load and parse a CSV file with TTI annotations"""
        df = pd.read_csv(csv_file)

        # Column mapping based on the CSV structure
        # Column 5 (index 4): TTI Frame Second File name (frame number)
        # Column 7 (index 6): Yolo TTI (0=TN, 1=TP, 2=FN, 3=FP)
        # Column 8 (index 7): Ground Truth Location of TTI (0=no TTI, 1=ALOS, 2=BLOS)

        return df

    def extract_binary_labels(self, df: pd.DataFrame) -> Tuple[List, List]:
        """
        Extract binary ground truth and predictions from DataFrame

        YOLO TTI meanings:
        0 = TN (True Negative): Model correctly predicted NO TTI
        1 = TP (True Positive): Model correctly predicted TTI
        2 = FN (False Negative): Model missed a TTI
        3 = FP (False Positive): Model incorrectly predicted TTI

        Ground Truth Location:
        0 = no TTI (TN/FP cases)
        1 = ALOS (TTI present)
        2 = BLOS (TTI present)

        Returns:
            ground_truth: Binary list (0=no TTI, 1=TTI present)
            predictions: Binary list (0=no TTI predicted, 1=TTI predicted)
        """
        ground_truth = []
        predictions = []

        # Column indices (0-based)
        yolo_tti_col = 6  # "Yolo TTI" column
        gt_location_col = 7  # "Ground Truth Location of TTI" column

        for idx, row in df.iterrows():
            try:
                # Get YOLO prediction
                yolo_value = row.iloc[yolo_tti_col]

                # Skip rows with missing or invalid data
                if pd.isna(yolo_value) or yolo_value == "?":
                    continue

                yolo_value = int(yolo_value)

                # Get ground truth
                gt_value = row.iloc[gt_location_col]
                if pd.isna(gt_value):
                    continue

                gt_value = int(gt_value)

                # Convert to binary
                # Ground truth: 0 = no TTI, 1 or 2 = TTI present
                gt_binary = 1 if gt_value > 0 else 0

                # Prediction: TP(1) or FP(3) = TTI predicted, TN(0) or FN(2) = no TTI predicted
                pred_binary = 1 if yolo_value in [1, 3] else 0

                ground_truth.append(gt_binary)
                predictions.append(pred_binary)

            except (ValueError, IndexError):
                continue

        return ground_truth, predictions

    def analyze_single_video(self, csv_file: str) -> Dict:
        """Analyze a single video CSV file"""
        video_name = Path(csv_file).stem
        print(f"Analyzing: {video_name}")

        df = self.load_csv_data(csv_file)
        ground_truth, predictions = self.extract_binary_labels(df)

        if not ground_truth or not predictions:
            print(f"  Warning: No valid data found in {video_name}")
            return None

        # Calculate confusion matrix with explicit labels to handle edge cases
        cm = confusion_matrix(ground_truth, predictions, labels=[0, 1])
        tn, fp, fn, tp = cm.ravel()

        # Calculate metrics
        accuracy = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) > 0 else 0
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
        f1_score = (
            2 * (precision * recall) / (precision + recall)
            if (precision + recall) > 0
            else 0
        )

        # Determine video type
        video_type = "safe" if "LapChol" in video_name else "bdi"

        result = {
            "video_name": video_name,
            "video_type": video_type,
            "ground_truth": ground_truth,
            "predictions": predictions,
            "tp": int(tp),
            "tn": int(tn),
            "fp": int(fp),
            "fn": int(fn),
            "accuracy": float(accuracy),
            "precision": float(precision),
            "recall": float(recall),
            "specificity": float(specificity),
            "f1_score": float(f1_score),
            "total_samples": len(ground_truth),
        }

        print(f"  Samples: {len(ground_truth)}")
        print(f"  TP: {tp}, TN: {tn}, FP: {fp}, FN: {fn}")
        print(
            f"  Accuracy: {accuracy:.3f}, Precision: {precision:.3f}, Recall: {recall:.3f}"
        )

        return result

    def run_analysis(self):
        """Run analysis on all CSV files"""
        print("Starting ROC analysis...")
        print("=" * 70)

        csv_files = glob.glob(str(self.csv_dir / "*.csv"))
        csv_files = [f for f in csv_files if Path(f).stem not in ["master-list"]]

        if not csv_files:
            print(f"No CSV files found in {self.csv_dir}")
            return []

        print(f"Found {len(csv_files)} CSV files to analyze\n")

        for csv_file in sorted(csv_files):
            result = self.analyze_single_video(csv_file)
            if result:
                self.results.append(result)
            print()

        return self.results

    def generate_roc_curves(self):
        """Generate ROC curves for overall, safe, and BDI videos"""
        if not self.results:
            print("No results to plot")
            return

        # Aggregate data
        all_gt = []
        all_pred = []
        safe_gt = []
        safe_pred = []
        bdi_gt = []
        bdi_pred = []

        for result in self.results:
            all_gt.extend(result["ground_truth"])
            all_pred.extend(result["predictions"])

            if result["video_type"] == "safe":
                safe_gt.extend(result["ground_truth"])
                safe_pred.extend(result["predictions"])
            else:
                bdi_gt.extend(result["ground_truth"])
                bdi_pred.extend(result["predictions"])

        # Create figure with ROC curves
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

        # Plot 1: Overall ROC curve
        if len(set(all_gt)) > 1:  # Need both classes for ROC
            fpr, tpr, thresholds = roc_curve(all_gt, all_pred)
            roc_auc = auc(fpr, tpr)

            ax1.plot(
                fpr,
                tpr,
                color="darkorange",
                lw=2,
                label=f"Overall ROC (AUC = {roc_auc:.3f})",
            )
        ax1.plot([0, 1], [0, 1], color="navy", lw=2, linestyle="--", label="Chance")
        ax1.set_xlim([0.0, 1.0])
        ax1.set_ylim([0.0, 1.05])
        ax1.set_xlabel("False Positive Rate", fontsize=12)
        ax1.set_ylabel("True Positive Rate", fontsize=12)
        ax1.set_title("Overall ROC Curve", fontsize=14, fontweight="bold")
        ax1.legend(loc="lower right")
        ax1.grid(True, alpha=0.3)

        # Plot 2: Safe vs BDI ROC curves
        if len(set(safe_gt)) > 1:
            fpr_safe, tpr_safe, _ = roc_curve(safe_gt, safe_pred)
            roc_auc_safe = auc(fpr_safe, tpr_safe)
            ax2.plot(
                fpr_safe,
                tpr_safe,
                color="blue",
                lw=2,
                label=f"Safe Videos (AUC = {roc_auc_safe:.3f})",
            )

        if len(set(bdi_gt)) > 1:
            fpr_bdi, tpr_bdi, _ = roc_curve(bdi_gt, bdi_pred)
            roc_auc_bdi = auc(fpr_bdi, tpr_bdi)
            ax2.plot(
                fpr_bdi,
                tpr_bdi,
                color="red",
                lw=2,
                label=f"BDI Videos (AUC = {roc_auc_bdi:.3f})",
            )

        ax2.plot([0, 1], [0, 1], color="navy", lw=2, linestyle="--", label="Chance")
        ax2.set_xlim([0.0, 1.0])
        ax2.set_ylim([0.0, 1.05])
        ax2.set_xlabel("False Positive Rate", fontsize=12)
        ax2.set_ylabel("True Positive Rate", fontsize=12)
        ax2.set_title("ROC Curves by Video Type", fontsize=14, fontweight="bold")
        ax2.legend(loc="lower right")
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()

        # Save figure
        roc_file = f"{self.output_prefix}_roc_curves.png"
        plt.savefig(roc_file, dpi=300, bbox_inches="tight")
        print(f"Saved ROC curves to: {roc_file}")
        plt.show()

        return roc_file

    def generate_confusion_matrices(self):
        """Generate visualization of confusion matrices (raw counts, row-normalized, column-normalized)"""
        if not self.results:
            return

        # Aggregate confusion matrices
        overall_cm = np.zeros((2, 2), dtype=int)
        safe_cm = np.zeros((2, 2), dtype=int)
        bdi_cm = np.zeros((2, 2), dtype=int)

        for result in self.results:
            cm = np.array([[result["tn"], result["fp"]], [result["fn"], result["tp"]]])
            overall_cm += cm

            if result["video_type"] == "safe":
                safe_cm += cm
            else:
                bdi_cm += cm

        # Create visualization with 3 rows: raw counts, row-normalized, column-normalized
        fig = plt.figure(figsize=(18, 15))

        # Helper function to plot confusion matrix
        def plot_cm(ax, cm, title, normalize=None):
            if normalize == "row":
                # Row normalization (recall per class)
                cm_norm = cm.astype("float") / (cm.sum(axis=1, keepdims=True) + 1e-10)
                display_cm = cm_norm
                fmt_str = ".2%"
                cmap = plt.cm.Blues
            elif normalize == "col":
                # Column normalization (precision per class)
                cm_norm = cm.astype("float") / (cm.sum(axis=0, keepdims=True) + 1e-10)
                display_cm = cm_norm
                fmt_str = ".2%"
                cmap = plt.cm.Greens
            else:
                # Raw counts
                display_cm = cm
                fmt_str = "d"
                cmap = plt.cm.Blues

            im = ax.imshow(display_cm, interpolation="nearest", cmap=cmap)
            ax.figure.colorbar(im, ax=ax)
            ax.set(
                xticks=np.arange(cm.shape[1]),
                yticks=np.arange(cm.shape[0]),
                xticklabels=["No TTI", "TTI"],
                yticklabels=["No TTI", "TTI"],
                title=title,
                ylabel="True label",
                xlabel="Predicted label",
            )

            # Add text annotations
            thresh = display_cm.max() / 2.0
            for i in range(cm.shape[0]):
                for j in range(cm.shape[1]):
                    if normalize:
                        # Show percentage for normalized matrices
                        text = f"{display_cm[i, j]:.1%}"
                    else:
                        # Show count and overall percentage for raw matrix
                        count = cm[i, j]
                        total = cm.sum()
                        percentage = (count / total * 100) if total > 0 else 0
                        text = f"{count}\n({percentage:.1f}%)"

                    ax.text(
                        j,
                        i,
                        text,
                        ha="center",
                        va="center",
                        color="white" if display_cm[i, j] > thresh else "black",
                        fontsize=12,
                        fontweight="bold",
                    )

        # Row 1: Raw counts
        ax1 = plt.subplot(3, 3, 1)
        plot_cm(ax1, overall_cm, "Overall - Raw Counts", normalize=None)

        ax2 = plt.subplot(3, 3, 2)
        plot_cm(ax2, safe_cm, "Safe Videos - Raw Counts", normalize=None)

        ax3 = plt.subplot(3, 3, 3)
        plot_cm(ax3, bdi_cm, "BDI Videos - Raw Counts", normalize=None)

        # Row 2: Row-normalized (Recall/Sensitivity per class)
        ax4 = plt.subplot(3, 3, 4)
        plot_cm(ax4, overall_cm, "Overall - Row Normalized (Recall)", normalize="row")

        ax5 = plt.subplot(3, 3, 5)
        plot_cm(ax5, safe_cm, "Safe Videos - Row Normalized (Recall)", normalize="row")

        ax6 = plt.subplot(3, 3, 6)
        plot_cm(ax6, bdi_cm, "BDI Videos - Row Normalized (Recall)", normalize="row")

        # Row 3: Column-normalized (Precision/PPV per class)
        ax7 = plt.subplot(3, 3, 7)
        plot_cm(
            ax7, overall_cm, "Overall - Column Normalized (Precision)", normalize="col"
        )

        ax8 = plt.subplot(3, 3, 8)
        plot_cm(
            ax8, safe_cm, "Safe Videos - Column Normalized (Precision)", normalize="col"
        )

        ax9 = plt.subplot(3, 3, 9)
        plot_cm(
            ax9, bdi_cm, "BDI Videos - Column Normalized (Precision)", normalize="col"
        )

        plt.tight_layout()

        cm_file = f"{self.output_prefix}_confusion_matrices.png"
        plt.savefig(cm_file, dpi=300, bbox_inches="tight")
        print(f"Saved confusion matrices to: {cm_file}")
        plt.show()

        return cm_file

    def generate_summary_report(self):
        """Generate summary statistics report"""
        if not self.results:
            print("No results to summarize")
            return None

        print("\n" + "=" * 80)
        print("ROC ANALYSIS SUMMARY REPORT")
        print("=" * 80)

        # Overall statistics
        safe_results = [r for r in self.results if r["video_type"] == "safe"]
        bdi_results = [r for r in self.results if r["video_type"] == "bdi"]

        # Calculate aggregate metrics
        def calc_aggregate_metrics(results):
            total_tp = sum(r["tp"] for r in results)
            total_tn = sum(r["tn"] for r in results)
            total_fp = sum(r["fp"] for r in results)
            total_fn = sum(r["fn"] for r in results)

            total = total_tp + total_tn + total_fp + total_fn
            accuracy = (total_tp + total_tn) / total if total > 0 else 0
            precision = (
                total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0
            )
            recall = (
                total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0
            )
            specificity = (
                total_tn / (total_tn + total_fp) if (total_tn + total_fp) > 0 else 0
            )
            f1 = (
                2 * (precision * recall) / (precision + recall)
                if (precision + recall) > 0
                else 0
            )

            return {
                "tp": total_tp,
                "tn": total_tn,
                "fp": total_fp,
                "fn": total_fn,
                "accuracy": accuracy,
                "precision": precision,
                "recall": recall,
                "specificity": specificity,
                "f1_score": f1,
                "total_samples": total,
            }

        overall_metrics = calc_aggregate_metrics(self.results)
        safe_metrics = calc_aggregate_metrics(safe_results) if safe_results else None
        bdi_metrics = calc_aggregate_metrics(bdi_results) if bdi_results else None

        # Print summary
        print(f"\nTotal Videos Analyzed: {len(self.results)}")
        print(f"  Safe Videos: {len(safe_results)}")
        print(f"  BDI Videos: {len(bdi_results)}")

        print("\n" + "-" * 80)
        print("OVERALL METRICS")
        print("-" * 80)
        print(f"Total Samples: {overall_metrics['total_samples']}")
        print(
            f"TP: {overall_metrics['tp']}, TN: {overall_metrics['tn']}, "
            f"FP: {overall_metrics['fp']}, FN: {overall_metrics['fn']}"
        )
        print(f"Accuracy:    {overall_metrics['accuracy']:.3f}")
        print(f"Precision:   {overall_metrics['precision']:.3f}")
        print(f"Recall:      {overall_metrics['recall']:.3f}")
        print(f"Specificity: {overall_metrics['specificity']:.3f}")
        print(f"F1 Score:    {overall_metrics['f1_score']:.3f}")

        if safe_metrics:
            print("\n" + "-" * 80)
            print("SAFE VIDEOS METRICS")
            print("-" * 80)
            print(f"Total Samples: {safe_metrics['total_samples']}")
            print(
                f"TP: {safe_metrics['tp']}, TN: {safe_metrics['tn']}, "
                f"FP: {safe_metrics['fp']}, FN: {safe_metrics['fn']}"
            )
            print(f"Accuracy:    {safe_metrics['accuracy']:.3f}")
            print(f"Precision:   {safe_metrics['precision']:.3f}")
            print(f"Recall:      {safe_metrics['recall']:.3f}")
            print(f"Specificity: {safe_metrics['specificity']:.3f}")
            print(f"F1 Score:    {safe_metrics['f1_score']:.3f}")

        if bdi_metrics:
            print("\n" + "-" * 80)
            print("BDI VIDEOS METRICS")
            print("-" * 80)
            print(f"Total Samples: {bdi_metrics['total_samples']}")
            print(
                f"TP: {bdi_metrics['tp']}, TN: {bdi_metrics['tn']}, "
                f"FP: {bdi_metrics['fp']}, FN: {bdi_metrics['fn']}"
            )
            print(f"Accuracy:    {bdi_metrics['accuracy']:.3f}")
            print(f"Precision:   {bdi_metrics['precision']:.3f}")
            print(f"Recall:      {bdi_metrics['recall']:.3f}")
            print(f"Specificity: {bdi_metrics['specificity']:.3f}")
            print(f"F1 Score:    {bdi_metrics['f1_score']:.3f}")

        # Create summary dictionary
        summary = {
            "overall": overall_metrics,
            "safe": safe_metrics,
            "bdi": bdi_metrics,
            "per_video": [
                {
                    "video_name": r["video_name"],
                    "video_type": r["video_type"],
                    "accuracy": r["accuracy"],
                    "precision": r["precision"],
                    "recall": r["recall"],
                    "f1_score": r["f1_score"],
                }
                for r in self.results
            ],
        }

        # Save to CSV
        summary_df = pd.DataFrame(
            [
                {
                    "Video": r["video_name"],
                    "Type": r["video_type"],
                    "Samples": r["total_samples"],
                    "TP": r["tp"],
                    "TN": r["tn"],
                    "FP": r["fp"],
                    "FN": r["fn"],
                    "Accuracy": r["accuracy"],
                    "Precision": r["precision"],
                    "Recall": r["recall"],
                    "Specificity": r["specificity"],
                    "F1_Score": r["f1_score"],
                }
                for r in self.results
            ]
        )

        csv_file = f"{self.output_prefix}_summary.csv"
        summary_df.to_csv(csv_file, index=False)
        print(f"\nSaved summary CSV to: {csv_file}")

        return summary


def main():
    """Main execution function"""
    parser = argparse.ArgumentParser(
        description="TTI ROC Analysis",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python roc_analysis.py --csv-dir frame_extraction
  python roc_analysis.py -c frame_extraction -o project_roc
        """,
    )

    parser.add_argument(
        "--csv-dir",
        "-c",
        required=True,
        help="Path to directory containing CSV files with TTI annotations",
    )

    parser.add_argument(
        "--output-prefix",
        "-o",
        default="roc_analysis",
        help="Prefix for output files (default: 'roc_analysis')",
    )

    args = parser.parse_args()

    # Validate path exists
    if not os.path.exists(args.csv_dir):
        print(f"Error: CSV directory does not exist: {args.csv_dir}")
        return None

    print(f"CSV directory: {args.csv_dir}")
    print(f"Output prefix: {args.output_prefix}")
    print()

    # Initialize analyzer
    analyzer = TTIROCAnalyzer(args.csv_dir, args.output_prefix)

    # Run analysis
    results = analyzer.run_analysis()

    if not results:
        print("No results generated. Check your CSV files.")
        return None

    # Generate outputs
    analyzer.generate_summary_report()
    analyzer.generate_roc_curves()
    analyzer.generate_confusion_matrices()

    print("\n" + "=" * 80)
    print("ROC ANALYSIS COMPLETE!")
    print("=" * 80)
    print("Generated files:")
    print(f"  - {args.output_prefix}_roc_curves.png")
    print(f"  - {args.output_prefix}_confusion_matrices.png")
    print(f"  - {args.output_prefix}_summary.csv")

    return results


if __name__ == "__main__":
    main()

