#!/bin/bash

# Define directories
INF_DIR="/Volumes/LORENZ/uhn/tti/inferences"
GNG_DIR="/Volumes/LORENZ/uhn/tti/gng_predictions_no_background"
OUT_DIR="./"  # or change this if you want outputs elsewhere

# Loop through each GNG video
for video in "$GNG_DIR"/*.mp4; do
    # Extract base name, e.g., lapchol_case_0001_03 from lapchol_case_0001_03.mp4
    base=$(basename "$video" .mp4)

    # Expected filtered JSON file path
    json_filtered="$INF_DIR/pred_${base}_filtered.json"

    # Check if filtered JSON exists
    if [ -f "$json_filtered" ]; then
        echo "Processing $base ..."
        python analyze_gng.py \
            --filtered_json "$json_filtered" \
            --gng_video "$video" \
            --output "${OUT_DIR}/gng_${base}.json"
    else
        echo "⚠️ Skipping $base — filtered JSON not found: $json_filtered"
    fi
done

echo "✅ All done!"
