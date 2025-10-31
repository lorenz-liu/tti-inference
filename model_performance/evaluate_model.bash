#!/bin/bash
#SBATCH -A madanigroup_gpu
#SBATCH -p gpu
#SBATCH --gres=gpu:v100:1
#SBATCH -t 3-00:00:00
#SBATCH --mem=16G
#SBATCH -c 8
#SBATCH -J model_performance
#SBATCH -o %x-%j.out
#SBATCH -e %x-%j.err
# ==============================================================================
# Batch-process frame images for TTI evaluation.
#
# This script finds all .jpg frame images in the specified base directory,
# and runs the optimized_eval.py script on each one, saving the results
# into a structured output directory.
# ==============================================================================
#
#
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

python
