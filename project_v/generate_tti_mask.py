#!/usr/bin/env python3
"""
Helper script to generate TTI segmentation mask from a single frame
Returns a binary mask PNG where white pixels indicate TTI regions
"""

import sys
import json
import cv2
import numpy as np

def main():
    if len(sys.argv) != 3:
        print("Usage: python generate_tti_mask.py <json_results> <output_mask>", file=sys.stderr)
        sys.exit(1)

    json_path = sys.argv[1]
    output_mask_path = sys.argv[2]

    # Read JSON results
    try:
        with open(json_path, 'r') as f:
            results = json.load(f)
    except Exception as e:
        print(f"ERROR: Could not read JSON file: {e}", file=sys.stderr)
        sys.exit(1)

    # Extract frame dimensions and detections
    try:
        data_units = results[0]["data_units"]
        data_hash = list(data_units.keys())[0]
        width = data_units[data_hash]["width"]
        height = data_units[data_hash]["height"]
        labels = data_units[data_hash]["labels"]

        # Get first frame's detections
        if not labels:
            print("WARNING: No detections found in JSON", file=sys.stderr)
            # Create empty mask
            mask = np.zeros((height, width), dtype=np.uint8)
            cv2.imwrite(output_mask_path, mask)
            sys.exit(0)

        frame_key = list(labels.keys())[0]
        frame_data = labels[frame_key]
        objects = frame_data.get("objects", [])

        # Create mask image
        mask = np.zeros((height, width), dtype=np.uint8)

        # Draw bounding boxes for TTI detections (tti_classification == 1)
        for obj in objects:
            if obj.get("tti_classification") == 1:
                bbox = obj["tti_bounding_box"]  # Use tissue interaction bbox
                x = int(bbox["x"] * width)
                y = int(bbox["y"] * height)
                w = int(bbox["w"] * width)
                h = int(bbox["h"] * height)

                # Fill the bounding box with white
                cv2.rectangle(mask, (x, y), (x + w, y + h), 255, -1)

        # Save mask
        cv2.imwrite(output_mask_path, mask)
        print(f"Mask saved to: {output_mask_path}")

    except Exception as e:
        print(f"ERROR: Failed to generate mask: {e}", file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    main()
