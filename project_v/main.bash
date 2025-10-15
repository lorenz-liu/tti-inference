#!/bin/bash
# ==========================================================
# Project V: TTI + GNG Overlap Analysis Automation Script
# Author: Lorenz
# Date: October 2025
# ==========================================================
# This script performs the following steps automatically:
# 1. Extract matching frames from GNG prediction videos.
# 2. Run TTI inference on extracted RGB frames.
# 3. Compute overlap between TTI predictions and GNG masks.
# 4. Save Go/No-Go percentages and classifications to CSV.
# ==========================================================

# ---------------------------
# CONFIGURATION
# ---------------------------
FRAME_DIR="/cluster/projects/madanigroup/lorenz/tti/frame_extraction"
GNG_DIR="/cluster/projects/madanigroup/lorenz/tti/gng_predictions_no_background"
TTI_MODEL_SCRIPT="/cluster/projects/madanigroup/lorenz/tti/optimized_eval.py"
OUTPUT_CSV="projectV_overlap_results.csv"
EXTRACT_SEC=20  # second of interest (change if needed)

# ---------------------------
# CSV HEADER
# ---------------------------
echo "video_name,frame_name,go_percent,nogo_percent,classification" > "$OUTPUT_CSV"

# ---------------------------
# MAIN LOOP THROUGH VIDEOS
# ---------------------------
for rgb_frame in "$FRAME_DIR"/*/frame_sec_*.jpg; do
    # Parse identifiers
    frame_name=$(basename "$rgb_frame")
    video_name=$(basename "$(dirname "$rgb_frame")")
    gng_video="$GNG_DIR/${video_name}.MP4"

    echo "Processing $video_name - $frame_name"

    # --------------------------------------
    # STEP 1: Extract GNG frame at target second
    # --------------------------------------
    python3 - <<EOF
import cv2
cap = cv2.VideoCapture(r"$gng_video")
fps = cap.get(cv2.CAP_PROP_FPS)
cap.set(cv2.CAP_PROP_POS_FRAMES, int(fps * $EXTRACT_SEC))
ret, frame = cap.read()
if ret:
    cv2.imwrite("gng_${frame_name}", frame)
cap.release()
EOF

    # --------------------------------------
    # STEP 2: Run TTI inference on RGB frame
    # --------------------------------------
    python3 "$TTI_MODEL_SCRIPT" --input "$rgb_frame" --output "tti_${frame_name%.jpg}.png"

    # --------------------------------------
    # STEP 3: Compute overlap and classification
    # --------------------------------------
    python3 - <<EOF
import cv2, numpy as np, csv

# Load masks
gng = cv2.imread("gng_${frame_name}")
tti = cv2.imread("tti_${frame_name%.jpg}.png")

# Resize to match dimensions
if gng.shape[:2] != tti.shape[:2]:
    tti = cv2.resize(tti, (gng.shape[1], gng.shape[0]))

# Create masks
go_mask   = cv2.inRange(gng,  (0,100,0),  (100,255,100))     # green
nogo_mask = cv2.inRange(gng,  (0,0,100),  (100,100,255))     # red
tti_mask  = cv2.inRange(tti,  (1,1,1),    (255,255,255))     # non-black

# Calculate overlaps
total_tti = np.count_nonzero(tti_mask)
go_overlap = np.count_nonzero(np.logical_and(tti_mask>0, go_mask>0))
nogo_overlap = np.count_nonzero(np.logical_and(tti_mask>0, nogo_mask>0))

if total_tti == 0:
    go_pct, nogo_pct = 0, 0
else:
    go_pct = go_overlap / total_tti * 100
    nogo_pct = nogo_overlap / total_tti * 100

# Classification rule
if go_pct >= 50:
    zone = "Go"
elif nogo_pct >= 50:
    zone = "No-Go"
else:
    zone = "Background"

# Append to CSV
with open("$OUTPUT_CSV", "a", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["$video_name", "$frame_name", f"{go_pct:.2f}", f"{nogo_pct:.2f}", zone])
EOF

    # Cleanup temporary GNG/TTI frames if desired
    rm -f "gng_${frame_name}" "tti_${frame_name%.jpg}.png"

done

# ---------------------------
# COMPLETION MESSAGE
# ---------------------------
echo "✅ Project V analysis complete. Results saved to: $OUTPUT_CSV"
