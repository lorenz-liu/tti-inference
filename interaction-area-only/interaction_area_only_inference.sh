#!/bin/bash

# Title: Run Batch Video Inference
# Description: Loops through all .mp4 files in the current directory,
# constructs the absolute path for each, and runs the specified Python script.

# --- Configuration ---
PYTHON_SCRIPT="/cluster/home/t130371uhn/github/tti-inference/interaction-area-only/interaction_area_only_inference.py"
# Get the absolute path of the directory where this script is executed
VIDEO_DIR="/cluster/projects/madanigroup/lorenz/tti/videos"
# ---------------------

echo "Starting batch inference using $PYTHON_SCRIPT"
echo "Targeting videos in: $VIDEO_DIR"
echo "----------------------------------------"

# Loop over all files ending with .mp4 in the current directory
for VIDEO_FILE in *.mp4; do
    # Check if the file actually exists and is not just the literal string "*.mp4"
    # (which happens if no .mp4 files are present)
    if [[ -f "$VIDEO_FILE" ]]; then

        # Construct the full absolute path as required by your example
        FULL_VIDEO_PATH="${VIDEO_DIR}/${VIDEO_FILE}"

        echo "-> Processing: $VIDEO_FILE"

        # Run the Python script
        python "$PYTHON_SCRIPT" --video "$FULL_VIDEO_PATH"

        # Check the exit status of the last command (the python script)
        if [ $? -eq 0 ]; then
            echo "   [SUCCESS] $VIDEO_FILE completed."
        else
            echo "   [ERROR] $VIDEO_FILE failed!"
        fi
        echo "" # Blank line for spacing
    fi
done

echo "----------------------------------------"
echo "Batch inference complete."
