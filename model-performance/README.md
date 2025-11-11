# TTI (Tool-Tissue Interaction) Detection Project

## Project Overview

This project implements a comprehensive system for detecting tool-tissue interactions in surgical videos using computer vision and deep learning. The system combines YOLO segmentation for tool/tissue detection with Vision Transformer (ViT) classification for TTI detection.

## Architecture

The system consists of two main components:
1. **YOLO Segmentation Model**: Detects surgical tools and tissues in video frames
2. **ViT Classifier**: Classifies tool-tissue interactions (TTI) using 5-channel input (RGB + Depth + Mask)

## Project Structure

```
TTI_Detection/
├── evaluate_vit_tti.py            # Main evaluation script
├── Model.py                       # Neural network architectures
├── training_ViT_pipe.py           # ViT training pipeline
├── create_dataset.py              # Dataset creation utilities
├── run.sh                         # Training script
├── inference.sh                   # Inference script
├── download_vit_model.py          # ViT model download
├── download_depth_model.py        # Depth model download
├── requirements.txt               # Python dependencies
├── README.md                      # This file
├── Videos/                        # Sample videos for testing
├── video_dataset/                 # Original video dataset used for YOLO training
├── processed_dataset_balanced/    # Processed video dataset used for TTI Classifier Training
├── model_folder/                  # Trained models
│   ├── ViT/                       # ViT classifier models
│   ├── EffNet_B0/                 # EfficientNet-B0 models
│   ├── EffNet_B1/                 # EfficientNet-B1 models
│   ├── EffNet_B2/                 # EfficientNet-B2 models
│   └── EffNet_B3/                 # EfficientNet-B3 models
├── runs_YOLO_M/                   # YOLO segmentation models
├── pretrained_models/             # Pre-trained ViT models
├── dpt_large_snapshot/            # Depth estimation models
├── ultralytics_yolo/              # YOLO framework
├── yolo_diagnosis_test/           # YOLO testing outputs
└── runs_OLD_DATASET/              # Old training runs
```

## Installation

### 1. Environment Setup

```bash
# Create virtual environment
python -m venv tti_env
source tti_env/bin/activate  # On Windows: tti_env\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Download Pre-trained Models

Run the following scripts to download required models:

```bash
# Download ViT model
python download_vit_model.py

# Download depth estimation model
python download_depth_model.py
```

## Training Pipeline

### 1. Dataset Preparation

The training pipeline expects data in the following format:

```
processed_dataset_balanced/
├── train/
│   ├── video001.pkl
│   ├── video002.pkl
│   └── ...
├── val/
│   ├── video003.pkl
│   ├── video004.pkl
│   └── ...
└── test/
    ├── video005.pkl
    ├── video006.pkl
    └── ...
```

Each `.pkl` file contains preprocessed frame data with 5-channel input (RGB + Depth + Mask).

### 2. YOLO Training

```bash
# Train YOLO segmentation model
python -m ultralytics_yolo.ultralytics yolo segment train \
    data=dataset.yaml \
    model=yolov8n-seg.pt \
    epochs=100 \
    imgsz=640 \
    batch=16 \
    device=0
```

### 3. ViT Classifier Training

Use the provided shell scripts for training:
```bash
# Training script
bash run.sh
```

OR 
```bash
# Train ViT classifier
python training_ViT_pipe.py \
    --epochs 40 \
    --batch_size 32 \
    --lr 1e-4 \
    --device cuda
```

## 4. Evaluation/Inference

Use the provided inference script:
```bash
bash inference.sh
```

OR
```bash
# Run evaluation on a video
python evaluate_vit_tti.py \
    --video "Videos/LapChol Case 0012 03.MP4" \
    --output "Results/evaluation_results.json" \
    --output_video "Results/annotated_video.mp4" \
    --show_heatmap \
    --vit_model "./model_folder/ViT/best_model.pt" \
    --yolo_model "./runs_YOLO_M/segment/train/weights/best.pt"
```

## Key Files and Their Functions

### Core Model Files

- **`Model.py`**: Contains neural network architectures (ViT, ResNet classifiers)
- **`evaluate_vit_tti.py`**: Main evaluation script with improved labeling
- **`training_ViT_pipe.py`**: ViT classifier training pipeline
- **`create_dataset.py`**: Dataset creation and preprocessing utilities

### Training Scripts

- **`run.sh`**: Main training script (SLURM compatible)
- **`inference.sh`**: Main inference script (SLURM compatible)
- **`training_NN_pipe.py`**: Alternative training pipeline

### Model Download Scripts

- **`download_vit_model.py`**: Downloads pre-trained ViT models
- **`download_depth_model.py`**: Downloads depth estimation models

## Model Architecture

### ViT Classifier
- **Input**: 5-channel data (RGB + Depth + Mask)
- **Architecture**: Vision Transformer (ViT-Base)
- **Output**: Binary classification (TTI vs No-TTI)
- **Features**: 768-dimensional embeddings

### YOLO Segmentation
- **Model**: YOLOv8 with segmentation head
- **Classes**: 12 tools + 9 tissue interactions
- **Output**: Segmentation masks and bounding boxes

## Configuration

### Model Paths
- ViT Model: `./model_folder/ViT/best_model.pt`
- YOLO Model: `./runs_YOLO_M/segment/train/weights/best.pt`
- Depth Model: `./dpt_large_snapshot/`

### Training Parameters
- **Epochs**: 40
- **Batch Size**: 32
- **Learning Rate**: 1e-4
- **Device**: CUDA (if available)

## Output Format

### JSON Results
```json
{
  "label_hash": "uuid",
  "data_units": {
    "data_hash": {
      "labels": {
        "frame_0": {
          "objects": [
            {
              "name": "TTI",
              "confidence": 0.85,
              "boundingBox": {"x": 0.1, "y": 0.2, "w": 0.3, "h": 0.4},
              "tool_info": {"name": "dissector", "class": 1},
              "tissue_info": {"name": "coagulation", "class": 13}
            }
          ]
        }
      }
    }
  }
}
```

### Annotated Video
- Green labels for each TTI detection
- Blue overlays for tool masks
- Green overlays for tissue masks
- Multiple labels per frame with smart positioning

## Troubleshooting

### Common Issues

1. **CUDA Out of Memory**
   - Reduce batch size in training scripts
   - Use smaller input resolution

2. **Model Loading Errors**
   - Ensure all model files are downloaded
   - Check file paths in scripts

3. **Dependency Issues**
   - Use the provided requirements.txt
   - Install CUDA toolkit if using GPU

### Performance Optimization

- Use GPU for training and inference
- Adjust batch size based on available memory
- Use frame sampling for long videos