#!/bin/bash

# Directory containing all the JSONs
INF_DIR="/Volumes/LORENZ/uhn/tti/inferences"

# Loop through all JSON files that are not *_filtered.json
for json_file in "$INF_DIR"/*.json; do
    # Skip filtered JSONs
    if [[ "$json_file" == *"_filtered.json" ]]; then
        continue
    fi

    echo "Processing $json_file ..."
    python select_frames.py --json "$json_file"
done

echo "✅ All select_frames.py runs complete!"
