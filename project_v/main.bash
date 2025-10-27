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
VIDEO_FILTER_REGEX="${VIDEO_FILTER_REGEX:-}"

echo "video_name,frame_name,second,go_percent,nogo_percent,classification" > "$OUTPUT_CSV"

# ==========================================================
# Helper: run TTI on a single frame using OpenCV only
# ==========================================================
run_tti_inference() {
    local img="$1"
    local out="$2"
    local tmpvid="__temp_video__.mp4"
    local tmpjson="${out%.png}.json"
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

    # Use absolute paths for output to ensure files are saved in the current directory
    local abs_json="$(pwd)/$tmpjson"
    local abs_out="$(pwd)/$out"

    # Run TTI model to generate JSON results
    python3 "$TTI_MODEL_SCRIPT" \
        --video "$tmpvid" \
        --output "$abs_json" \
        --start_frame 0 --end_frame 1 --frame_step 1 \
        --device cuda

    # Convert JSON results to PNG mask
    python3 generate_tti_mask.py "$tmpjson" "$abs_out"

    rm -f "$tmpvid" "$tmpjson"
}

# ==========================================================
# Helper: resolve matching GNG video file (handles case, dashes, 30fps)
# ==========================================================
resolve_gng_video() {
    local video_name="$1"
    python3 - "$GNG_DIR" "$video_name" <<'EOF'
import os
import re
import sys

gng_dir, target = sys.argv[1], sys.argv[2]

def normalize(name: str) -> str:
    # Drop non-alphanumeric characters and lowercase
    return re.sub(r'[^a-z0-9]', '', name.lower())

target_norm = normalize(target)
if not target_norm:
    sys.exit(0)

candidates = []
for entry in os.listdir(gng_dir):
    path = os.path.join(gng_dir, entry)
    if not os.path.isfile(path):
        continue
    base, ext = os.path.splitext(entry)
    norm = normalize(base)
    if not norm:
        continue

    if norm == target_norm:
        base_penalty = 0
    elif norm.startswith(target_norm):
        base_penalty = len(norm) - len(target_norm)
    elif target_norm.startswith(norm):
        base_penalty = len(target_norm) - len(norm) + 5
    else:
        continue

    ext_penalty = {"mp4": 0, "mov": 1}.get(ext.lower().lstrip("."), 5)
    # Prefer 30fps assets when available
    fps_bonus = -1 if base.lower().endswith("30fps") else 0

    candidates.append(((base_penalty, ext_penalty, fps_bonus, entry.lower()), path))

if candidates:
    candidates.sort(key=lambda item: item[0])
    print(candidates[0][1])
EOF
}

# ==========================================================
# MAIN LOOP
# ==========================================================
for folder in "$FRAME_DIR"/*/; do
    video_name=$(basename "$folder")
    if [[ -n "$VIDEO_FILTER_REGEX" ]] && [[ ! "$video_name" =~ $VIDEO_FILTER_REGEX ]]; then
        continue
    fi

    gng_video="$(resolve_gng_video "$video_name")"

    if [[ -z "$gng_video" || ! -f "$gng_video" ]]; then
        echo "⚠️  Skipping $video_name (no matching GNG video found)"
        continue
    fi

    echo "Processing video: $video_name"
    echo "  • Using GNG video: $gng_video"

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

        # --------------------------------------
        # STEP 3: Compute overlap + classification
        # --------------------------------------
        python3 - <<EOF
import cv2, numpy as np, csv, sys

gng = cv2.imread("gng_${frame_name}")
tti = cv2.imread("tti_${frame_name%.jpg}.png", cv2.IMREAD_GRAYSCALE)

if gng is None:
    print("ERROR: Could not read GNG frame: gng_${frame_name}", file=sys.stderr)
    sys.exit(1)
if tti is None:
    print("ERROR: Could not read TTI mask: tti_${frame_name%.jpg}.png", file=sys.stderr)
    sys.exit(1)

# Resize TTI mask to match GNG frame dimensions
if gng.shape[:2] != tti.shape[:2]:
    tti = cv2.resize(tti, (gng.shape[1], gng.shape[0]))

# Extract GNG zone masks (green=Go, red=NoGo)
go_mask   = cv2.inRange(gng,  (0,100,0),  (100,255,100))
nogo_mask = cv2.inRange(gng,  (0,0,100),  (100,100,255))

# TTI mask is already binary (white pixels = TTI regions)
tti_mask = tti

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
