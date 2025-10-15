#!/bin/bash
#SBATCH -A madanigroup_gpu
#SBATCH -p gpu
#SBATCH --gres=gpu:v100:1
#SBATCH -t 3-00:00:00
#SBATCH --mem=16G
#SBATCH -c 8
#SBATCH -J project_v
#SBATCH -o %x-%j.out
#SBATCH -e %x-%j.err

# Initialize conda
echo "=== Initializing conda ==="
source ~/.bashrc
if [ $? -eq 0 ]; then
    echo "SUCCESS: Bashrc sourced"
else
    echo "ERROR: Failed to source bashrc"
fi

# Check if conda is available
echo "=== Checking conda availability ==="
which conda
if [ $? -eq 0 ]; then
    echo "SUCCESS: Conda found at $(which conda)"
else
    echo "ERROR: Conda not found, trying alternative initialization..."
    # Try alternative conda initialization paths
    if [ -f ~/miniconda3/etc/profile.d/conda.sh ]; then
        source ~/miniconda3/etc/profile.d/conda.sh
        echo "SUCCESS: Conda initialized from miniconda3"
    elif [ -f ~/anaconda3/etc/profile.d/conda.sh ]; then
        source ~/anaconda3/etc/profile.d/conda.sh
        echo "SUCCESS: Conda initialized from anaconda3"
    else
        echo "ERROR: Could not find conda initialization script"
        exit 1
    fi
fi

# Activate conda environment
echo "=== Activating conda environment 'tti' ==="
conda activate tti
if [ $? -eq 0 ]; then
    echo "SUCCESS: Conda environment 'tti' activated"
    echo "Active environment: $CONDA_DEFAULT_ENV"
    echo "Python location: $(which python)"
    echo "Python version: $(python --version)"
else
    echo "ERROR: Failed to activate conda environment 'tti'"
    echo "Available environments:"
    conda env list
    exit 1
fi

# ==========================================================
# Project V: ACCELERATED TTI + GNG Overlap Analysis
# Author: Lorenz (Optimized for GPU + 8 CPUs)
# ==========================================================
# Optimizations:
# - Batch GPU inference (processes multiple frames at once)
# - Parallel frame extraction and overlap computation (8 workers)
# - Reduced subprocess overhead with consolidated Python script
# ==========================================================

FRAME_DIR="/cluster/projects/madanigroup/lorenz/tti/frame_extraction"
GNG_DIR="/cluster/projects/madanigroup/lorenz/tti/gng_predictions_no_background"
TTI_MODEL_SCRIPT="../optimized_eval.py"
OUTPUT_CSV="project_v_overlap_results.csv"
BATCH_SIZE=16  # Process 16 frames at once on GPU
NUM_WORKERS=8   # Use all 8 CPUs for parallel processing

echo "=== Starting accelerated processing ==="
echo "GPU Batch Size: $BATCH_SIZE"
echo "CPU Workers: $NUM_WORKERS"

# Initialize output CSV
echo "video_name,frame_name,second,go_percent,nogo_percent,classification" > "$OUTPUT_CSV"

# ==========================================================
# ACCELERATED PYTHON PROCESSING SCRIPT
# ==========================================================
python3 - <<'PYTHON_SCRIPT'
import os
import sys
import cv2
import numpy as np
import csv
import torch
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
from collections import defaultdict
import subprocess
import tempfile
import shutil

# Configuration from environment
FRAME_DIR = os.environ.get("FRAME_DIR", "/cluster/projects/madanigroup/lorenz/tti/frame_extraction")
GNG_DIR = os.environ.get("GNG_DIR", "/cluster/projects/madanigroup/lorenz/tti/gng_predictions_no_background")
TTI_MODEL_SCRIPT = os.environ.get("TTI_MODEL_SCRIPT", "../optimized_eval.py")
OUTPUT_CSV = os.environ.get("OUTPUT_CSV", "project_v_overlap_results.csv")
BATCH_SIZE = int(os.environ.get("BATCH_SIZE", 16))
NUM_WORKERS = int(os.environ.get("NUM_WORKERS", 8))

def find_gng_video(video_name, gng_dir):
    """Find matching GNG video for a given video name."""
    # Try exact match first
    gng_path = os.path.join(gng_dir, f"{video_name.replace('_', ' ')}.MP4")
    if os.path.exists(gng_path):
        return gng_path

    # Try case-insensitive search
    gng_dir_path = Path(gng_dir)
    if gng_dir_path.exists():
        for file in gng_dir_path.iterdir():
            if file.name.lower() == f"{video_name.replace('_', ' ')}.mp4".lower():
                return str(file)
    return None

def extract_gng_frames_batch(gng_video_path, frame_seconds):
    """Extract multiple GNG frames at once from a video."""
    cap = cv2.VideoCapture(gng_video_path)
    if not cap.isOpened():
        return {}

    fps = cap.get(cv2.CAP_PROP_FPS)
    gng_frames = {}

    for sec in sorted(set(frame_seconds)):
        frame_num = int(fps * sec)
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
        ret, frame = cap.read()
        if ret:
            gng_frames[sec] = frame

    cap.release()
    return gng_frames

def create_batch_video(frame_paths, output_video_path):
    """Create a temporary video from multiple frames for batch inference."""
    if not frame_paths:
        return False

    # Read first frame to get dimensions
    first_frame = cv2.imread(frame_paths[0])
    if first_frame is None:
        return False

    h, w = first_frame.shape[:2]
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(output_video_path, fourcc, 1, (w, h))

    for path in frame_paths:
        frame = cv2.imread(path)
        if frame is not None:
            writer.write(frame)

    writer.release()
    return True

def run_tti_batch_inference(video_path, output_dir, num_frames):
    """Run TTI inference on a batch video."""
    try:
        cmd = [
            "python3", TTI_MODEL_SCRIPT,
            "--video", video_path,
            "--output", output_dir,
            "--start_frame", "0",
            "--end_frame", str(num_frames),
            "--frame_step", "1",
            "--device", "cuda"
        ]
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
        return result.returncode == 0
    except Exception as e:
        print(f"Error running TTI inference: {e}", file=sys.stderr)
        return False

def compute_overlap(gng_frame, tti_frame):
    """Compute overlap between GNG and TTI frames."""
    if gng_frame is None or tti_frame is None:
        return 0.0, 0.0, "Background"

    # Resize TTI to match GNG if needed
    if gng_frame.shape[:2] != tti_frame.shape[:2]:
        tti_frame = cv2.resize(tti_frame, (gng_frame.shape[1], gng_frame.shape[0]))

    # Create masks
    go_mask = cv2.inRange(gng_frame, (0, 100, 0), (100, 255, 100))
    nogo_mask = cv2.inRange(gng_frame, (0, 0, 100), (100, 100, 255))
    tti_mask = cv2.inRange(tti_frame, (1, 1, 1), (255, 255, 255))

    # Compute overlaps
    total_tti = np.count_nonzero(tti_mask)
    if total_tti == 0:
        return 0.0, 0.0, "Background"

    go_overlap = np.count_nonzero(np.logical_and(tti_mask > 0, go_mask > 0))
    nogo_overlap = np.count_nonzero(np.logical_and(tti_mask > 0, nogo_mask > 0))

    go_pct = (go_overlap / total_tti) * 100
    nogo_pct = (nogo_overlap / total_tti) * 100

    # Classify
    if go_pct >= 50:
        classification = "Go"
    elif nogo_pct >= 50:
        classification = "No-Go"
    else:
        classification = "Background"

    return go_pct, nogo_pct, classification

def process_video_batch(video_name, frame_dir, gng_dir):
    """Process all frames for a single video using batch processing."""
    print(f"Processing video: {video_name}")

    # Find video folder
    video_folder = os.path.join(frame_dir, video_name)
    if not os.path.exists(video_folder):
        return []

    # Find matching GNG video
    gng_video = find_gng_video(video_name, gng_dir)
    if not gng_video:
        print(f"⚠️  Skipping {video_name} (no matching GNG video found)")
        return []

    # Collect all frames
    frame_files = sorted([f for f in os.listdir(video_folder) if f.startswith("frame_sec_") and f.endswith(".jpg")])
    if not frame_files:
        return []

    # Extract frame info
    frame_info = []
    for frame_file in frame_files:
        frame_path = os.path.join(video_folder, frame_file)
        # Extract second from filename
        sec_str = frame_file.replace("frame_sec_", "").replace(".jpg", "").lstrip("0") or "0"
        try:
            sec = int(sec_str)
            frame_info.append((frame_path, frame_file, sec))
        except ValueError:
            continue

    if not frame_info:
        return []

    # Extract all GNG frames at once
    frame_seconds = [sec for _, _, sec in frame_info]
    gng_frames = extract_gng_frames_batch(gng_video, frame_seconds)

    results = []

    # Process in batches
    for batch_start in range(0, len(frame_info), BATCH_SIZE):
        batch_end = min(batch_start + BATCH_SIZE, len(frame_info))
        batch = frame_info[batch_start:batch_end]

        print(f"  Processing batch {batch_start//BATCH_SIZE + 1}/{(len(frame_info)-1)//BATCH_SIZE + 1} ({len(batch)} frames)")

        # Create temporary directory for this batch
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create batch video
            batch_video = os.path.join(temp_dir, "batch_video.mp4")
            frame_paths = [fp for fp, _, _ in batch]

            if not create_batch_video(frame_paths, batch_video):
                print(f"  ⚠️  Failed to create batch video")
                continue

            # Run TTI inference on batch
            tti_output_dir = os.path.join(temp_dir, "tti_output")
            os.makedirs(tti_output_dir, exist_ok=True)

            if not run_tti_batch_inference(batch_video, tti_output_dir, len(batch)):
                print(f"  ⚠️  TTI inference failed for batch")
                continue

            # Process results
            for idx, (frame_path, frame_file, sec) in enumerate(batch):
                # Find TTI output (could be PNG or other format)
                tti_output = None
                for ext in ['.png', '.jpg', '.jpeg']:
                    potential_path = os.path.join(tti_output_dir, f"frame_{idx:04d}{ext}")
                    if os.path.exists(potential_path):
                        tti_output = potential_path
                        break

                if tti_output is None:
                    print(f"    ⚠️  No TTI output for {frame_file}")
                    continue

                # Get GNG frame
                gng_frame = gng_frames.get(sec)
                tti_frame = cv2.imread(tti_output)

                # Compute overlap
                go_pct, nogo_pct, classification = compute_overlap(gng_frame, tti_frame)

                results.append({
                    'video_name': video_name,
                    'frame_name': frame_file,
                    'second': sec,
                    'go_percent': f"{go_pct:.2f}",
                    'nogo_percent': f"{nogo_pct:.2f}",
                    'classification': classification
                })

    print(f"✅ Completed {video_name}: {len(results)} frames processed")
    return results

# Main processing
def main():
    # Find all video folders
    video_folders = [d for d in os.listdir(FRAME_DIR)
                    if os.path.isdir(os.path.join(FRAME_DIR, d))]

    print(f"Found {len(video_folders)} videos to process")

    all_results = []

    # Process videos in parallel (CPU-bound operations)
    with ProcessPoolExecutor(max_workers=min(NUM_WORKERS, len(video_folders))) as executor:
        futures = {executor.submit(process_video_batch, video_name, FRAME_DIR, GNG_DIR): video_name
                  for video_name in video_folders}

        for future in as_completed(futures):
            video_name = futures[future]
            try:
                results = future.result()
                all_results.extend(results)
            except Exception as e:
                print(f"❌ Error processing {video_name}: {e}", file=sys.stderr)

    # Write all results to CSV
    if all_results:
        with open(OUTPUT_CSV, 'a', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=['video_name', 'frame_name', 'second',
                                                   'go_percent', 'nogo_percent', 'classification'])
            for result in all_results:
                writer.writerow(result)

    print(f"\n✅ All processing complete!")
    print(f"Total frames processed: {len(all_results)}")
    print(f"Results saved to: {OUTPUT_CSV}")

if __name__ == "__main__":
    main()
PYTHON_SCRIPT

echo "✅ Processing complete. Results saved to: $OUTPUT_CSV"
