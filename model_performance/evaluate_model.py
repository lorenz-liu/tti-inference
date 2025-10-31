#!/usr/bin/env python3
"""
Comprehensive model evaluation script for TTI detection.
Supports both ViT and EfficientNet models with automatic model type detection.
Saves results in appropriate model-specific directories.
"""

import torch
import numpy as np
import os
import pickle
import json
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import argparse
from datetime import datetime
import torchvision.models as models
import torch.nn as nn
import torch.nn.functional as F

# Import model architectures
from Model import ROIClassifierViT, ROIClassifierViTNoDepth
from transformers import ViTModel, ViTConfig

# --- Configuration ---
DEFAULT_MODEL_PATH = "/cluster/projects/madanigroup/lorenz/tti/vit.pt"
DEFAULT_DATA_PATH = "/cluster/projects/madanigroup/lorenz/tti/legacy/test_vit_dataset"
BATCH_SIZE = 32
CLASS_NAMES = ["No-TTI", "TTI"]

# Plotting Configuration
CM_FIG_SIZE = (10, 8)
CM_COLOR_MAP = "Blues"
METRICS_FIG_SIZE = (12, 6)
METRICS_NAMES = ["Accuracy", "Sensitivity", "Specificity", "Precision", "F1-Score"]
METRICS_COLORS = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd"]

# Visualization Configuration
NUM_VIS_SAMPLES = 12
VIS_ROWS = 3
VIS_COLS = 4
VIS_FIG_SIZE = (16, 12)


# EfficientNet model classes (imported directly to avoid circular imports)
class ROIClassifierEfficientNet(nn.Module):
    """EfficientNet-based classifier for TTI detection"""

    def __init__(self, num_classes, efficientnet_version="b0"):
        super().__init__()

        # Channel reduction layers - same pattern as ViT version
        self.first = nn.Conv2d(5, 4, kernel_size=1, stride=1, padding=0, bias=False)
        self.pre_conv = nn.Conv2d(4, 3, kernel_size=1, stride=1, padding=0, bias=False)

        # EfficientNet backbone - load pretrained model
        if efficientnet_version == "b0":
            self.backbone = models.efficientnet_b0(weights="IMAGENET1K_V1")
            feature_dim = 1280  # EfficientNet-B0 feature dimension
        elif efficientnet_version == "b1":
            self.backbone = models.efficientnet_b1(weights="IMAGENET1K_V1")
            feature_dim = 1280  # EfficientNet-B1 feature dimension
        elif efficientnet_version == "b2":
            self.backbone = models.efficientnet_b2(weights="IMAGENET1K_V1")
            feature_dim = 1408  # EfficientNet-B2 feature dimension
        elif efficientnet_version == "b3":
            self.backbone = models.efficientnet_b3(weights="IMAGENET1K_V1")
            feature_dim = 1536  # EfficientNet-B3 feature dimension
        else:
            raise ValueError(
                f"Unsupported EfficientNet version: {efficientnet_version}"
            )

        # Replace classifier with identity to get features
        self.backbone.classifier = nn.Identity()

        # Custom classification head
        self.fc = nn.Linear(feature_dim, num_classes)

        # Store version for reference
        self.efficientnet_version = efficientnet_version

    def forward(self, x):
        # Channel reduction: 5 -> 4 -> 3 channels
        x = self.first(x)
        x = self.pre_conv(x)

        # EfficientNet feature extraction
        features = self.backbone(x)

        # Classification
        out = self.fc(features)
        return out


class ROIClassifierEfficientNetNoDepth(nn.Module):
    """EfficientNet-based classifier without depth channel"""

    def __init__(self, num_classes, efficientnet_version="b0"):
        super().__init__()

        # Channel reduction layer - 4 -> 3 channels
        self.pre_conv = nn.Conv2d(4, 3, kernel_size=1, stride=1, padding=0, bias=False)

        # EfficientNet backbone
        if efficientnet_version == "b0":
            self.backbone = models.efficientnet_b0(weights="IMAGENET1K_V1")
            feature_dim = 1280
        elif efficientnet_version == "b1":
            self.backbone = models.efficientnet_b1(weights="IMAGENET1K_V1")
            feature_dim = 1280
        elif efficientnet_version == "b2":
            self.backbone = models.efficientnet_b2(weights="IMAGENET1K_V1")
            feature_dim = 1408
        elif efficientnet_version == "b3":
            self.backbone = models.efficientnet_b3(weights="IMAGENET1K_V1")
            feature_dim = 1536
        else:
            raise ValueError(
                f"Unsupported EfficientNet version: {efficientnet_version}"
            )

        self.backbone.classifier = nn.Identity()
        self.fc = nn.Linear(feature_dim, num_classes)
        self.efficientnet_version = efficientnet_version

    def forward(self, x):
        # Channel reduction: 4 -> 3 channels
        x = self.pre_conv(x)

        # EfficientNet feature extraction
        features = self.backbone(x)

        # Classification
        out = self.fc(features)
        return out


def detect_model_info(model_path):
    """Detect model type and configuration from path"""
    model_dir = os.path.dirname(model_path)

    # Check if there's a model_config.json file
    config_path = os.path.join(model_dir, "model_config.json")
    if os.path.exists(config_path):
        try:
            with open(config_path, "r") as f:
                config = json.load(f)

            # Ensure required keys exist, add missing ones
            if "model_type" not in config:
                # Try to detect from path if model_type is missing
                config = detect_from_path(model_path)
            elif "model_name" not in config:
                # Add model_name if missing
                if config.get("model_type") == "ViT":
                    config["model_name"] = "ViT"
                elif config.get("model_type") == "EfficientNet":
                    version = config.get("version", "b0").upper()
                    config["model_name"] = f"EfficientNet-{version}"
                else:
                    config["model_name"] = config.get("model_type", "Unknown")

            return config
        except (json.JSONDecodeError, KeyError, Exception) as e:
            print(f"Warning: Could not read model config file: {e}")
            # Fall back to path-based detection
            return detect_from_path(model_path)

    # Fallback: detect from path
    return detect_from_path(model_path)


def detect_from_path(model_path):
    """Detect model info from file path"""
    if "ViT" in model_path:
        return {
            "model_type": "ViT",
            "model_name": "ViT",
            "no_depth": "NoDepth" in model_path,
        }
    elif "EffNet" in model_path:
        # Extract EfficientNet version from path
        version = "b0"  # default
        if "EffNet_B0" in model_path:
            version = "b0"
        elif "EffNet_B1" in model_path:
            version = "b1"
        elif "EffNet_B2" in model_path:
            version = "b2"
        elif "EffNet_B3" in model_path:
            version = "b3"

        return {
            "model_type": "EfficientNet",
            "model_name": f"EfficientNet-{version.upper()}",
            "version": version,
            "no_depth": "NoDepth" in model_path,
        }

    # Default to ViT for backward compatibility
    return {"model_type": "ViT", "model_name": "ViT", "no_depth": False}


def load_test_data(data_path):
    """Loads all .pkl files from the specified test data directory."""
    print(f"Loading test data from: {data_path}")
    if not os.path.exists(data_path):
        print(f"Error: Data directory not found at '{data_path}'")
        return None, None

    pkl_files = [f for f in os.listdir(data_path) if f.endswith(".pkl")]
    if not pkl_files:
        print(f"Error: No .pkl files found in '{data_path}'")
        return None, None

    all_frames = []
    all_labels = []
    for pkl_file in tqdm(pkl_files, desc="Loading .pkl files"):
        file_path = os.path.join(data_path, pkl_file)
        with open(file_path, "rb") as f:
            data = pickle.load(f)
            for item in data:
                all_frames.append(item["frame_data"])
                all_labels.append(item["label"])

    print(f"Loaded {len(all_frames)} total test samples.")
    return np.array(all_frames), np.array(all_labels)


def create_dataloader(frames, labels):
    """Creates a DataLoader for the test set."""
    if frames is None or labels is None:
        return None

    frames_tensor = torch.FloatTensor(frames)
    labels_tensor = torch.LongTensor(labels)

    dataset = TensorDataset(frames_tensor, labels_tensor)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False)

    print(f"Test DataLoader created with {len(loader)} batches.")
    return loader


def load_model(model_path, device):
    """Loads the trained PyTorch model with automatic type detection."""
    print(f"Loading model from: {model_path}")
    if not os.path.exists(model_path):
        print(f"Error: Model file not found at '{model_path}'")
        return None, None

    try:
        # Detect model type and configuration
        model_info = detect_model_info(model_path)
        print(f"Detected model: {model_info['model_name']}")

        # Initialize the appropriate model architecture
        if model_info["model_type"] == "ViT":
                        if model_info.get('no_depth', False):
                            model = ROIClassifierViTNoDepth(num_hoi_classes=2)
                        else:
                            model = ROIClassifierViT(num_hoi_classes=2)        elif model_info["model_type"] == "EfficientNet":
            version = model_info.get("version", "b0")
            if model_info.get("no_depth", False):
                model = ROIClassifierEfficientNetNoDepth(
                    num_classes=2, efficientnet_version=version
                )
            else:
                model = ROIClassifierEfficientNet(
                    num_classes=2, efficientnet_version=version
                )
        else:
            raise ValueError(f"Unsupported model type: {model_info['model_type']}")

        # Load the saved state dictionary
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.to(device)
        model.eval()  # Set the model to evaluation mode
        print("Model loaded successfully and set to evaluation mode.")
        return model, model_info
    except Exception as e:
        print(f"Error loading model: {e}")
        return None, None


def evaluate(model, loader, device, model_info):
    """Runs the evaluation loop and returns predictions and true labels."""
    all_preds = []
    all_labels = []
    all_probs = []

    with torch.no_grad():
        for frames, labels in tqdm(loader, desc="Evaluating model"):
            frames = frames.to(device)

            outputs = model(frames)

            # Handle different output formats
            if model_info["model_type"] == "ViT":
                # ViT models output sigmoids
                probs = outputs.cpu().numpy()
                predicted = (probs > 0.5).astype(int)
                if len(predicted.shape) > 1:
                    predicted = (
                        predicted[:, 1] if predicted.shape[1] > 1 else predicted[:, 0]
                    )
                    probs = probs[:, 1] if probs.shape[1] > 1 else probs[:, 0]
            else:
                # EfficientNet models output logits
                probs = torch.softmax(outputs, dim=1)
                _, predicted = torch.max(outputs.data, 1)
                predicted = predicted.cpu().numpy()
                probs = probs[:, 1].cpu().numpy()  # Probability of TTI class

            all_preds.extend(predicted)
            all_labels.extend(labels.numpy())
            all_probs.extend(probs)

    return np.array(all_preds), np.array(all_labels), np.array(all_probs)


def calculate_metrics(preds, labels, probs):
    """Calculate comprehensive evaluation metrics."""
    accuracy = accuracy_score(labels, preds)
    report = classification_report(
        labels, preds, target_names=CLASS_NAMES, output_dict=True
    )
    cm = confusion_matrix(labels, preds)

    # Additional metrics
    tn, fp, fn, tp = cm.ravel()
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = (
        2 * (precision * recall) / (precision + recall)
        if (precision + recall) > 0
        else 0
    )

    metrics = {
        "accuracy": accuracy,
        "sensitivity": sensitivity,
        "specificity": specificity,
        "precision": precision,
        "recall": recall,
        "f1_score": f1,
        "confusion_matrix": cm.tolist(),
        "classification_report": report,
        "test_samples": len(labels),
        "tti_samples": int(np.sum(labels == 1)),
        "no_tti_samples": int(np.sum(labels == 0)),
    }

    return metrics


def save_results(metrics, model_info, model_path, save_dir):
    """Save comprehensive evaluation results."""
    os.makedirs(save_dir, exist_ok=True)

    # Create evaluation results dictionary
    results = {
        "model_info": model_info,
        "model_path": model_path,
        "evaluation_date": datetime.now().isoformat(),
        "metrics": metrics,
    }

    # Save results as JSON
    results_path = os.path.join(save_dir, "evaluation_results.json")
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)

    # Save summary text
    summary_path = os.path.join(save_dir, "evaluation_summary.txt")
    with open(summary_path, "w") as f:
        f.write("=== TTI Detection Model Evaluation Summary ===\n\n")
        f.write(f"Model: {model_info['model_name']}\n")
        f.write(f"Model Path: {model_path}\n")
        f.write(f"Evaluation Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")

        f.write("=== Test Dataset ===\n")
        f.write(f"Total Samples: {metrics['test_samples']}\n")
        f.write(f"TTI Samples: {metrics['tti_samples']}\n")
        f.write(f"No-TTI Samples: {metrics['no_tti_samples']}\n\n")

        f.write("=== Performance Metrics ===\n")
        f.write(
            f"Accuracy: {metrics['accuracy']:.4f} ({metrics['accuracy'] * 100:.2f}%)\n"
        )
        f.write(f"Sensitivity (Recall): {metrics['sensitivity']:.4f}\n")
        f.write(f"Specificity: {metrics['specificity']:.4f}\n")
        f.write(f"Precision: {metrics['precision']:.4f}\n")
        f.write(f"F1-Score: {metrics['f1_score']:.4f}\n\n")

        f.write("=== Confusion Matrix ===\n")
        f.write("           Predicted\n")
        f.write("         No-TTI  TTI\n")
        f.write(
            f"Actual No-TTI  {metrics['confusion_matrix'][0][0]:4d}  {metrics['confusion_matrix'][0][1]:4d}\n"
        )
        f.write(
            f"       TTI     {metrics['confusion_matrix'][1][0]:4d}  {metrics['confusion_matrix'][1][1]:4d}\n"
        )

    print(f"Results saved to: {save_dir}")
    return results_path, summary_path


def plot_results(metrics, model_info, save_dir):
    """Create and save evaluation plots."""
    # Set up the plotting style
    plt.style.use("default")

    # Create confusion matrix plot
    plt.figure(figsize=CM_FIG_SIZE)

    cm = np.array(metrics["confusion_matrix"])
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap=CM_COLOR_MAP,
        xticklabels=CLASS_NAMES,
        yticklabels=CLASS_NAMES,
        cbar_kws={"label": "Count"},
    )

    plt.title(
        f"Confusion Matrix - {model_info['model_name']}\n"
        f"Accuracy: {metrics['accuracy'] * 100:.2f}%",
        fontsize=14,
        fontweight="bold",
    )
    plt.xlabel("Predicted Label", fontsize=12)
    plt.ylabel("True Label", fontsize=12)

    # Add text annotations with percentages
    total = cm.sum()
    for i in range(2):
        for j in range(2):
            percentage = cm[i, j] / total * 100
            plt.text(
                j + 0.5,
                i + 0.3,
                f"({percentage:.1f}%)",
                ha="center",
                va="center",
                fontsize=10,
                color="gray",
            )

    plt.tight_layout()
    confusion_path = os.path.join(save_dir, "confusion_matrix.png")
    plt.savefig(confusion_path, dpi=300, bbox_inches="tight")
    plt.close()

    # Create metrics bar plot
    plt.figure(figsize=METRICS_FIG_SIZE)

    metrics_values = [
        metrics["accuracy"],
        metrics["sensitivity"],
        metrics["specificity"],
        metrics["precision"],
        metrics["f1_score"],
    ]

    bars = plt.bar(METRICS_NAMES, metrics_values, color=METRICS_COLORS)

    # Add value labels on bars
    for bar, value in zip(bars, metrics_values):
        plt.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.01,
            f"{value:.3f}",
            ha="center",
            va="bottom",
            fontweight="bold",
        )

    plt.title(
        f"Performance Metrics - {model_info['model_name']}",
        fontsize=14,
        fontweight="bold",
    )
    plt.ylabel("Score", fontsize=12)
    plt.ylim(0, 1.1)
    plt.grid(axis="y", alpha=0.3)
    plt.tight_layout()

    metrics_path = os.path.join(save_dir, "performance_metrics.png")
    plt.savefig(metrics_path, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"Plots saved to: {save_dir}")
    return confusion_path, metrics_path


def visualize_predictions(frames, labels, preds, model_info, save_dir):
    """Displays a grid of images with their true and predicted labels."""
    indices = np.random.choice(
        len(frames), size=min(NUM_VIS_SAMPLES, len(frames)), replace=False
    )

    # Create subplot grid
    fig, axes = plt.subplots(VIS_ROWS, VIS_COLS, figsize=VIS_FIG_SIZE)
    fig.suptitle(
        f"Sample Predictions - {model_info['model_name']}",
        fontsize=16,
        fontweight="bold",
    )

    for i, idx in enumerate(indices):
        row = i // VIS_COLS
        col = i % VIS_COLS
        ax = axes[row, col]

        # Get the RGB channels (first 3) and convert from CHW to HWC
        image_chw = frames[idx][:3]
        image_hwc = np.transpose(image_chw, (1, 2, 0))

        # Normalize for display
        image_hwc = (image_hwc - image_hwc.min()) / (image_hwc.max() - image_hwc.min())

        ax.imshow(image_hwc)
        ax.axis("off")

        true_label = CLASS_NAMES[labels[idx]]
        pred_label = CLASS_NAMES[preds[idx]]

        color = "green" if true_label == pred_label else "red"

        ax.set_title(
            f"True: {true_label}\nPred: {pred_label}", color=color, fontweight="bold"
        )

    # Hide empty subplots
    for i in range(len(indices), VIS_ROWS * VIS_COLS):
        row = i // VIS_COLS
        col = i % VIS_COLS
        axes[row, col].axis("off")

    plt.tight_layout()
    predictions_path = os.path.join(save_dir, "sample_predictions.png")
    plt.savefig(predictions_path, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"Sample predictions saved to: {predictions_path}")
    return predictions_path


def get_model_save_directory(model_path):
    """Get the appropriate save directory for the model."""
    model_dir = os.path.dirname(model_path)

    # If model is already in a model_folder subdirectory, use that
    if "model_folder" in model_dir:
        return model_dir

    # Otherwise, try to determine from model info
    model_info = detect_model_info(model_path)

    if model_info["model_type"] == "ViT":
        return os.path.join("./model_folder", "ViT")
    elif model_info["model_type"] == "EfficientNet":
        version = model_info.get("version", "b0").upper()
        folder_name = f"EffNet_{version}"
        if model_info.get("no_depth", False):
            folder_name += "_NoDepth"
        return os.path.join("./model_folder", folder_name)

    # Default fallback
    return "./model_folder"


def display_results(metrics, model_info):
    """Display evaluation results to console."""
    print("\n" + "=" * 60)
    print("           EVALUATION RESULTS")
    print("=" * 60)
    print(f"Model: {model_info['model_name']}")
    print(f"Test Samples: {metrics['test_samples']}")
    print(f"TTI Samples: {metrics['tti_samples']}")
    print(f"No-TTI Samples: {metrics['no_tti_samples']}")
    print("-" * 60)
    print(f"Accuracy: {metrics['accuracy']:.4f} ({metrics['accuracy'] * 100:.2f}%)")
    print(f"Sensitivity (Recall): {metrics['sensitivity']:.4f}")
    print(f"Specificity: {metrics['specificity']:.4f}")
    print(f"Precision: {metrics['precision']:.4f}")
    print(f"F1-Score: {metrics['f1_score']:.4f}")
    print("-" * 60)
    print("Confusion Matrix:")
    print("           Predicted")
    print("         No-TTI  TTI")
    print(
        f"Actual No-TTI  {metrics['confusion_matrix'][0][0]:4d}  {metrics['confusion_matrix'][0][1]:4d}"
    )
    print(
        f"       TTI     {metrics['confusion_matrix'][1][0]:4d}  {metrics['confusion_matrix'][1][1]:4d}"
    )
    print("=" * 60)


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate a trained TTI-classifier model."
    )
    parser.add_argument(
        "--model_path",
        type=str,
        default=DEFAULT_MODEL_PATH,
        help="Path to the trained model file (.pt)",
    )
    parser.add_argument(
        "--data_path",
        type=str,
        default=DEFAULT_DATA_PATH,
        help="Path to the test data directory",
    )
    parser.add_argument(
        "--save_dir",
        type=str,
        default="./",
        help="Directory to save results (auto-detected if not provided)",
    )
    args = parser.parse_args()

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 1. Load Data
    frames, labels = load_test_data(args.data_path)
    if frames is None:
        return

    # 2. Create DataLoader
    loader = create_dataloader(frames, labels)
    if loader is None:
        return

    # 3. Load Model
    model, model_info = load_model(args.model_path, device)
    if model is None:
        return

    # 4. Evaluate
    predictions, true_labels, probabilities = evaluate(
        model, loader, device, model_info
    )

    # 5. Calculate metrics
    metrics = calculate_metrics(predictions, true_labels, probabilities)

    # 6. Display results
    display_results(metrics, model_info)

    # 7. Determine save directory
    save_dir = args.save_dir

    # 8. Save results
    results_path, summary_path = save_results(
        metrics, model_info, args.model_path, save_dir
    )

    # 9. Create and save plots
    confusion_path, metrics_path = plot_results(metrics, model_info, save_dir)

    # 10. Visualize sample predictions
    predictions_path = visualize_predictions(
        frames, true_labels, predictions, model_info, save_dir
    )

    print(f"\n✅ Evaluation complete! Results saved to: {save_dir}")
    print(f"Summary: {summary_path}")
    print(f"Detailed results: {results_path}")
    print(f"Confusion matrix: {confusion_path}")
    print(f"Performance metrics: {metrics_path}")
    print(f"Sample predictions: {predictions_path}")
    print(f"\nAll files saved in: {save_dir}")


if __name__ == "__main__":
    main()
