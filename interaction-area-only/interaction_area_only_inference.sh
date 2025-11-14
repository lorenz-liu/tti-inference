#!/bin/bash
#SBATCH -A madanigroup_gpu
#SBATCH -p gpu
#SBATCH --gres=gpu:v100:1
#SBATCH -t 3-00:00:00
#SBATCH --mem=16G
#SBATCH -c 8
#SBATCH -J interaction-area-only
#SBATCH -o %x-%j.out
#SBATCH -e %x-%j.err
# ==============================================================================
# Batch-process video files for TTI evaluation.
# ==============================================================================

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

# --- Configuration (using your provided paths) ---
PYTHON_SCRIPT="/cluster/home/t130371uhn/interaction_area_only_inference.py"
VIDEO_DIR="/cluster/projects/madanigroup/lorenz/tti/videos"
# ---------------------

echo "Starting batch inference using $PYTHON_SCRIPT"
echo "Targeting videos in: $VIDEO_DIR"
echo "----------------------------------------"

# FIX: Loop over all files ending with .mp4 explicitly in the target directory
# We loop over the full path and extract the filename for display.
for FULL_PATH in "$VIDEO_DIR"/*.mp4; do
    # Check if the file actually exists and is not the literal string "*.mp4"
    if [[ -f "$FULL_PATH" ]]; then

        # Extract the base filename for display purposes
        VIDEO_FILE=$(basename "$FULL_PATH")

        # Set the path variable used in the python script call
        FULL_VIDEO_PATH="$FULL_PATH"

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
