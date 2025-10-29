# This script requires opencv-python and numpy.
# You can install them using: pip install opencv-python numpy

import argparse
import json
import os
import cv2
import numpy as np
import csv


def analyze_gng_zones(json_path, gng_video_path):
    """_summary_

    Args:
        json_path (_type_): _description_
        gng_video_path (_type_): _description_
    """
    # BGR color values derived from the user's hex codes
    # GO: #066913 -> RGB(6, 105, 19) -> BGR(19, 105, 6)
    # NOGO: #86160F -> RGB(134, 22, 15) -> BGR(15, 22, 134)
    GO_COLOR = np.array([19, 105, 6])
    NOGO_COLOR = np.array([15, 22, 134])

    # 1. Load JSON
    try:
        with open(json_path, "r") as f:
            data = json.load(f)
    except FileNotFoundError:
        print(f"Error: JSON file not found at {json_path}")
        return

    # 2. Extract frame numbers and boxes
    frames_to_process = {}
    for item in data:
        for data_unit in item.get("data_units", {}).values():
            for frame_num_str, label_data in data_unit.get("labels", {}).items():
                frame_num = int(frame_num_str)
                if label_data.get("objects"):
                    bbox = label_data["objects"][0].get("tti_bounding_box")
                    if bbox:
                        frames_to_process[frame_num] = bbox

    if not frames_to_process:
        print("No frames with 'tti_bounding_box' found in the JSON file.")
        return

    # 3. Open video
    cap = cv2.VideoCapture(gng_video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video file at {gng_video_path}")
        return

    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # 4. Setup CSV output
    output_csv_path = "gng_intersection_analysis.csv"
    with open(output_csv_path, "w", newline="") as csvfile:
        csv_writer = csv.writer(csvfile)
        csv_writer.writerow(["frame", "tti-go", "tti-nogo", "tti-void"])

        print(f"Analyzing frames and writing results to {output_csv_path}...")

        # 5. Process frames
        for frame_num, bbox_rel in sorted(frames_to_process.items()):
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
            ret, gng_frame = cap.read()

            if not ret:
                print(f"Warning: Could not read frame {frame_num}. Skipping.")
                continue

            # Convert relative bbox to absolute pixel coordinates
            x = int(bbox_rel["x"] * frame_width)
            y = int(bbox_rel["y"] * frame_height)
            w = int(bbox_rel["w"] * frame_width)
            h = int(bbox_rel["h"] * frame_height)

            # Ensure the bbox is within frame boundaries
            x = max(0, x)
            y = max(0, y)
            w = min(w, frame_width - x)
            h = min(h, frame_height - y)

            total_pixels_in_bbox = w * h
            if total_pixels_in_bbox == 0:
                csv_writer.writerow([frame_num, "0%", "0%", "100%"])
                continue

            # Crop the region of interest (the bounding box area)
            bbox_roi = gng_frame[y : y + h, x : x + w]

            # Create masks for exact colors within the ROI
            go_mask = cv2.inRange(bbox_roi, GO_COLOR, GO_COLOR)
            nogo_mask = cv2.inRange(bbox_roi, NOGO_COLOR, NOGO_COLOR)

            # Count pixels
            go_pixels = cv2.countNonZero(go_mask)
            nogo_pixels = cv2.countNonZero(nogo_mask)
            void_pixels = total_pixels_in_bbox - go_pixels - nogo_pixels

            # Calculate percentages
            go_perc = (go_pixels / total_pixels_in_bbox) * 100
            nogo_perc = (nogo_pixels / total_pixels_in_bbox) * 100
            void_perc = (void_pixels / total_pixels_in_bbox) * 100

            # Write formatted percentages to CSV
            csv_writer.writerow(
                [frame_num, f"{go_perc:.0f}%", f"{nogo_perc:.0f}%", f"{void_perc:.0f}%"]
            )

    # 6. Cleanup
    cap.release()
    print("Done.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Analyze intersection of TTI bounding box with GO/NOGO zones."
    )
    parser.add_argument(
        "--filtered_json",
        required=True,
        help="Path to the filtered JSON file with bounding box data.",
    )
    parser.add_argument(
        "--gng_video", required=True, help="Path to the GO/NOGO video file."
    )
    args = parser.parse_args()

    analyze_gng_zones(args.filtered_json, args.gng_video)
