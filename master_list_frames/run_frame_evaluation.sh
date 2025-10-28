#!/bin/bash
#SBATCH -A madanigroup_gpu
#SBATCH -p gpu
#SBATCH --gres=gpu:v100:1
#SBATCH -t 3-00:00:00
#SBATCH --mem=16G
#SBATCH -c 8
#SBATCH -J frames_eval
#SBATCH -o %x-%j.out
#SBATCH -e %x-%j.err
# ==============================================================================
# Batch-process frame images for TTI evaluation.
#
# This script finds all .jpg frame images in the specified base directory,
# and runs the optimized_eval.py script on each one, saving the results
# into a structured output directory.
# ==============================================================================

set -e  # Exit immediately if a command exits with a non-zero status.

# --- Configuration ---
# Adjust these paths according to your environment.

# Path to the Python evaluation script.
# IMPORTANT: Use a specific path. The '~' shortcut might not expand correctly in all environments.
EVAL_SCRIPT_PATH="~/github/tti-inference/optimized_eval.py"

# Base directory containing the video frame folders (e.g., LapChol_Case_0001_03/frame_sec_000.jpg).
FRAMES_BASE_DIR="/cluster/projects/madanigroup/lorenz/tti/master_list_frames"

# Base directory where the output JSON and annotated images will be stored.
RESULTS_BASE_DIR="/cluster/projects/madanigroup/lorenz/tti/master_frames_eval"

# Check if the evaluation script exists
if [ ! -f "$EVAL_SCRIPT_PATH" ]; then
    echo "Error: Evaluation script not found at: $EVAL_SCRIPT_PATH"
    exit 1
fi

# Find all frame images and process them.
# The `find` command is robust for handling filenames with spaces or special characters.
find "$FRAMES_BASE_DIR" -type f -name "*.jpg" | while read -r image_path; do
    # Get the parent directory name (e.g., LapChol_Case_0001_03)
    video_name=$(basename "$(dirname "$image_path")")

    # Get the frame filename without the .jpg extension (e.g., frame_sec_000)
    frame_name=$(basename "$image_path" .jpg)

    # Define the output directory for this specific video's results.
    output_video_dir="$RESULTS_BASE_DIR/$video_name"

    # Create the video-specific output directory.
    mkdir -p "$output_video_dir"

    # Define the full paths for the output files.
    output_json_path="$output_video_dir/${frame_name}.json"
    output_image_path="$output_video_dir/${frame_name}_annotated.jpg"

    echo "--------------------------------------------------"
    echo "Processing:   $image_path"
    echo "Output JSON:  $output_json_path"
    echo "Output Image: $output_image_path"
    echo "--------------------------------------------------"

    # Execute the Python evaluation script.
    # We add --show_heatmap and --show_roi_box for better visual results.
    python "$EVAL_SCRIPT_PATH" \
      --image "$image_path" \
      --output "$output_json_path" \
      --output_image "$output_image_path" \
      --show_heatmap \
      --show_roi_box

done

echo ""
echo "All frames processed successfully."
