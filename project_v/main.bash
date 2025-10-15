#!/bin/bash
# ==========================================================
# Project V: Automated TTI + GNG Overlap Analysis for All Frames
# Author: Lorenz
# ==========================================================
# This script:
# 1. Iterates through every subfolder in frame_extraction/
# 2. For each extracted frame, gets the matching second (from filename)
# 3. Extracts the same frame from the GNG prediction video
# 4. Runs TTI inference (via single-frame wrapper)
# 5. Computes Go/No-Go overlap and classification
# 6. Appends results to a CSV summary file
# ==========================================================

# ---------------------------
# CONFIGURATION
# ---------------------------
FRAME_DIR="/cluster/projects/madanigroup/lorenz/tti/frame_extraction"
GNG_DIR="/cluster/projects/madanigroup/lorenz/tti/gng_predictions_no_background"
TTI_MODEL_SCRIPT="../optimized_eval.py"
OUTPUT_CSV="project_v_overlap_results.csv"

# ---------------------------
# CSV HEADER
# ---------------------------
echo "video_name,frame_name,second,go_percent,nogo_percent,classification" > "$OUTPUT_CSV"

# ==========================================================
# FUNCTION: Run TTI inference on a single frame
# (wraps optimized_eval.py which expects a video)
# ==========================================================
run_tti_inference() {
    local img="$1"
    local out="$2"
    local tmpvid="__temp_video__.mp4"
    local tmpdir="__temp_frames__"

    mkdir -p "$tmpdir"
    cp "$img" "$tmpdir/frame_0001.jpg"

    # Create a 1-frame video
    ffmpeg -y -framerate 1 -i "$tmpdir/frame_%04d.jpg" \
        -c:v libx264 -pix_fmt yuv420p "$tmpvid" -loglevel quiet

    # Run optimized_eval.py on the temp video
    python3 "$TTI_MODEL_SCRIPT" \
        --video "$tmpvid" \
        --output "$out" \
        --start_frame 0 --end_frame 1 --frame_step 1 \
        --device cuda

    # Cleanup
    rm -rf "$tmpdir" "$tmpvid"
}

# ==========================================================
# MAIN LOOP THROUGH VIDEO FOLDERS
# ==========================================================
for folder in "$FRAME_DIR"/*/; do
    video_name=$(basename "$folder")
    gng_video="$GNG_DIR/${video_name//_/' '}.MP4"

    # Skip if no corresponding GNG video
    if [[ ! -f "$gng_video" ]]; then
        echo "⚠️  Skipping $video_name (no matching GNG video found)"
        continue
    fi

    echo "Processing video: $video_name"

    # ---------------------------
    # LOOP THROUGH ALL FRAMES IN THIS VIDEO
    # ---------------------------
    for rgb_frame in "$folder"/frame_sec_*.jpg; do
        frame_name=$(basename "$rgb_frame")
        sec=$(echo "$frame_name" | grep -oP '(?<=frame_sec_)\d+')

        echo "  → Frame $frame_name (second=$sec)"

        # --------------------------------------
        # STEP 1: Extract corresponding GNG frame
        # --------------------------------------
        python3 - <<EOF
import cv2
cap = cv2.VideoCapture(r"$gng_video")
fps = cap.get(cv2.CAP_PROP_FPS)
sec_str = "$sec".lstrip("0") or "0"
sec_val = int(sec_str)
cap.set(cv2.CAP_PROP_POS_FRAMES, int(fps * sec_val))
ret, frame = cap.read()
if ret:
    cv2.imwrite("gng_${frame_name}", frame)
cap.release()
EOF

        # --------------------------------------
        # STEP 2: Run TTI inference (wrapper)
        # --------------------------------------
        run_tti_inference "$rgb_frame" "tti_${frame_name%.jpg}.png"

        # --------------------------------------
        # STEP 3: Compute overlap + classification
        # --------------------------------------
        python3 - <<EOF
import cv2, numpy as np, csv

gng = cv2.imread("gng_${frame_name}")
tti = cv2.imread("tti_${frame_name%.jpg}.png")
if gng is None or tti is None:
    exit()

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
EOF

        # --------------------------------------
        # CLEANUP
        # --------------------------------------
        rm -f "gng_${frame_name}" "tti_${frame_name%.jpg}.png"
    done
done

# ---------------------------
# COMPLETION MESSAGE
# ---------------------------
echo "✅ All frames processed. Results saved to: $OUTPUT_CSV"
