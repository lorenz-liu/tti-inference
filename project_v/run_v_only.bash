#!/bin/bash
#SBATCH -A madanigroup_gpu
#SBATCH -p gpu
#SBATCH --gres=gpu:v100:1
#SBATCH -t 3-00:00:00
#SBATCH --mem=16G
#SBATCH -c 8
#SBATCH -J project_v_bdi_only
#SBATCH -o %x-%j.out
#SBATCH -e %x-%j.err

# Helper wrapper to rerun main.bash only for videos that start with "V"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export VIDEO_FILTER_REGEX='^V'
bash "$SCRIPT_DIR/main.bash" "$@"
