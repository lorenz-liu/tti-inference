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

    # Target dimensions are based on the original video
    TARGET_WIDTH = 1280
    TARGET_HEIGHT = 720

    # 1. Load JSON
    try:
        with open(json_path, 'r') as f:
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

    gng_native_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    gng_native_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    needs_resize = (gng_native_height != TARGET_HEIGHT or gng_native_width != TARGET_WIDTH)
    if needs_resize:
        print(f"Go/No-Go video resolution ({gng_native_width}x{gng_native_height}) differs from target ({TARGET_WIDTH}x{TARGET_HEIGHT}). Frames will be resized and centered.")

    # 4. Setup CSV output
    output_csv_path = "gng_intersection_analysis.csv"
    with open(output_csv_path, 'w', newline='') as csvfile:
        csv_writer = csv.writer(csvfile)
        csv_writer.writerow(['frame', 'tti-go', 'tti-nogo', 'tti-void'])

        print(f"Analyzing frames and writing results to {output_csv_path}...")

        # 5. Process frames
        for frame_num, bbox_rel in sorted(frames_to_process.items()):
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
            ret, frame_from_video = cap.read()

            if not ret:
                print(f"Warning: Could not read frame {frame_num}. Skipping.")
                continue

            # Standardize the frame to the target resolution if needed
            final_gng_frame = None
            if needs_resize:
                aspect_ratio = gng_native_width / gng_native_height
                new_h = TARGET_HEIGHT
                new_w = int(new_h * aspect_ratio)
                
                resized = cv2.resize(frame_from_video, (new_w, new_h), interpolation=cv2.INTER_NEAREST)
                
                canvas = np.zeros((TARGET_HEIGHT, TARGET_WIDTH, 3), dtype=np.uint8)
                x_offset = (TARGET_WIDTH - new_w) // 2
                canvas[:, x_offset:x_offset+new_w] = resized
                
                final_gng_frame = canvas
            else:
                final_gng_frame = frame_from_video

            # BBox coordinates are relative to the TARGET dimensions
            x = int(bbox_rel['x'] * TARGET_WIDTH)
            y = int(bbox_rel['y'] * TARGET_HEIGHT)
            w = int(bbox_rel['w'] * TARGET_WIDTH)
            h = int(bbox_rel['h'] * TARGET_HEIGHT)

            # Ensure the bbox is within frame boundaries
            x = max(0, x)
            y = max(0, y)
            w = min(w, TARGET_WIDTH - x)
            h = min(h, TARGET_HEIGHT - y)

            total_pixels_in_bbox = w * h
            if total_pixels_in_bbox == 0:
                csv_writer.writerow([frame_num, '0%', '0%', '100%'])
                continue

            # Crop the region of interest from the standardized frame
            bbox_roi = final_gng_frame[y:y+h, x:x+w]

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
            csv_writer.writerow([
                frame_num,
                f"{go_perc:.0f}%",
                f"{nogo_perc:.0f}%",
                f"{void_perc:.0f}%"
            ])

    # 6. Cleanup
    cap.release()
    print("Done.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Analyze intersection of TTI bounding box with GO/NOGO zones.')
    parser.add_argument('--filtered_json', required=True, help='Path to the filtered JSON file with bounding box data.')
    parser.add_argument('--gng_video', required=True, help='Path to the GO/NOGO video file.')
    args = parser.parse_args()

    analyze_gng_zones(args.filtered_json, args.gng_video)