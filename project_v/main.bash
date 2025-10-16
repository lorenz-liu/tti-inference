#!/bin/bash
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
# Project V: Automated TTI + GNG Overlap Analysis for All Frames
# Author: Lorenz
# ==========================================================
# Dependencies: python3, cv2, numpy
# ==========================================================

FRAME_DIR="/cluster/projects/madanigroup/lorenz/tti/frame_extraction"
GNG_DIR="/cluster/projects/madanigroup/lorenz/tti/gng_predictions_no_background"
TTI_MODEL_SCRIPT="../optimized_eval.py"
OUTPUT_CSV="project_v_overlap_results.csv"

echo "video_name,frame_name,second,go_percent,nogo_percent,classification" > "$OUTPUT_CSV"

# ==========================================================
# Helper: run TTI on a single frame using OpenCV only
# ==========================================================
run_tti_inference() {
    local img="$1"
    local out="$2"
    local tmpvid="__temp_video__.mp4"
    local tmpcode="
import cv2
frame = cv2.imread(r'$img')
if frame is None:
    raise SystemExit('Cannot read frame: $img')
h, w = frame.shape[:2]
writer = cv2.VideoWriter(r'$tmpvid', cv2.VideoWriter_fourcc(*'mp4v'), 1, (w,h))
writer.write(frame)
writer.release()
"
    python3 -c "$tmpcode"

    # Use absolute path for output to ensure it's saved in the current directory
    local abs_out="$(pwd)/$out"
    python3 "$TTI_MODEL_SCRIPT" \
        --video "$tmpvid" \
        --output "$abs_out" \
        --start_frame 0 --end_frame 1 --frame_step 1 \
        --device cuda

    rm -f "$tmpvid"
}

# ==========================================================
# MAIN LOOP
# ==========================================================
for folder in "$FRAME_DIR"/*/; do
    video_name=$(basename "$folder")
    gng_video="$GNG_DIR/${video_name//_/' '}.MP4"

    if [[ ! -f "$gng_video" ]]; then
        echo "⚠️  Skipping $video_name (no matching GNG video found)"
        continue
    fi

    echo "Processing video: $video_name"

    for rgb_frame in "$folder"/frame_sec_*.jpg; do
        frame_name=$(basename "$rgb_frame")
        sec=$(echo "$frame_name" | grep -oP '(?<=frame_sec_)\d+')
        echo "  → Frame $frame_name (second=$sec)"

        # --------------------------------------
        # STEP 1: Extract GNG frame
        # --------------------------------------
        python3 - <<EOF
import cv2, sys
cap = cv2.VideoCapture(r"$gng_video")
if not cap.isOpened():
    print("ERROR: Could not open GNG video: $gng_video", file=sys.stderr)
    sys.exit(1)
fps = cap.get(cv2.CAP_PROP_FPS)
sec_str = "$sec".lstrip("0") or "0"
sec_val = int(sec_str)
cap.set(cv2.CAP_PROP_POS_FRAMES, int(fps * sec_val))
ret, frame = cap.read()
cap.release()
if not ret:
    print("ERROR: Could not read frame at second $sec from GNG video: $gng_video", file=sys.stderr)
    sys.exit(1)
cv2.imwrite("gng_${frame_name}", frame)
EOF

        # --------------------------------------
        # STEP 2: Run TTI inference (OpenCV wrapper)
        # --------------------------------------
        run_tti_inference "$rgb_frame" "tti_${frame_name%.jpg}.png"

        # Debug: Check if TTI output file exists and show its location
        if [[ -f "tti_${frame_name%.jpg}.png" ]]; then
            echo "    ✓ TTI output found: $(pwd)/tti_${frame_name%.jpg}.png"
        else
            echo "    ✗ TTI output NOT found in current directory: $(pwd)"
            echo "    Searching for TTI output file..."
            find . -name "tti_${frame_name%.jpg}.png" -type f 2>/dev/null || echo "    File not found anywhere in current tree"
        fi

        # --------------------------------------
        # STEP 3: Compute overlap + classification
        # --------------------------------------
        python3 - <<EOF
import cv2, numpy as np, csv, sys
import os

# Debug: Check file status before reading
gng_path = "gng_${frame_name}"
tti_path = "tti_${frame_name%.jpg}.png"

print(f"DEBUG: Checking GNG file: {gng_path}", file=sys.stderr)
if os.path.exists(gng_path):
    print(f"  - Exists: Yes, Size: {os.path.getsize(gng_path)} bytes", file=sys.stderr)
else:
    print(f"  - Exists: No", file=sys.stderr)

print(f"DEBUG: Checking TTI file: {tti_path}", file=sys.stderr)
if os.path.exists(tti_path):
    print(f"  - Exists: Yes, Size: {os.path.getsize(tti_path)} bytes", file=sys.stderr)
else:
    print(f"  - Exists: No", file=sys.stderr)

gng = cv2.imread(gng_path)
tti = cv2.imread(tti_path)

if gng is None:
    print(f"ERROR: Could not read GNG frame: {gng_path}", file=sys.stderr)
    print(f"  File exists but cv2.imread() returned None - file may be corrupted", file=sys.stderr)
    sys.exit(1)
if tti is None:
    print(f"ERROR: Could not read TTI frame: {tti_path}", file=sys.stderr)
    print(f"  File exists but cv2.imread() returned None - file may be corrupted", file=sys.stderr)
    sys.exit(1)

if gng.shape[:2] != tti.shape[:2]:
    tti = cv2.resize(tti, (gng.shape[1], gng.shape[0]))

go_mask   = cv2.inRange(gng,  (0,100,0),  (100,255,100))
nogo_mask = cv2.inRange(gng,  (0,0,100),  (100,100,255))
tti_mask  = cv2.inRange(tti,  (1,1,1),    (255,255,255))

total_tti = np.count_nonzero(tti_mask)
go_overlap = np.count_nonzero(np.logical_and(tti_mask>0, go_mask>0))
nogo_overlap = np.count_nonzero(np.logical_and(tti_mask>0, nogo_mask>0))

if total_tti == 0:
    go_pct, nogo_pct = 0, 0
else:
    go_pct = go_overlap / total_tti * 100
    nogo_pct = nogo_overlap / total_tti * 100

if go_pct >= 50:
    zone = "Go"
elif nogo_pct >= 50:
    zone = "No-Go"
else:
    zone = "Background"

with open("$OUTPUT_CSV", "a", newline="") as f:
    csv.writer(f).writerow(["$video_name", "$frame_name", "$sec", f"{go_pct:.2f}", f"{nogo_pct:.2f}", zone])
    print(f"✓ Wrote result: {zone} (Go: {go_pct:.1f}%, NoGo: {nogo_pct:.1f}%)")
EOF

        rm -f "gng_${frame_name}" "tti_${frame_name%.jpg}.png"
    done
done

echo "✅ All frames processed. Results saved to: $OUTPUT_CSV"
