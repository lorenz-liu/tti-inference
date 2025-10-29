# This script requires opencv-python and numpy.
# You can install them using: pip install opencv-python numpy

import argparse
import json
import os
import cv2
import numpy as np


def draw_bounding_boxes(video_path, json_path):
    """_summary_

    Args:
        video_path (_type_): _description_
        json_path (_type_): _description_
    """
    # 1. Load JSON
    try:
        with open(json_path, "r") as f:
            data = json.load(f)
    except FileNotFoundError:
        print(f"Error: JSON file not found at {json_path}")
        return
    except json.JSONDecodeError:
        print(f"Error: Could not decode JSON from {json_path}")
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
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video file at {video_path}")
        return

    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # 4. Create output directory
    output_dir = "frames_with_boxes"
    os.makedirs(output_dir, exist_ok=True)
    print(f"Saving output frames to '{output_dir}/'")

    # 5. Process frames
    for frame_num, bbox_rel in sorted(frames_to_process.items()):
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
        ret, frame = cap.read()

        if not ret:
            print(f"Warning: Could not read frame {frame_num}. Skipping.")
            continue

        # Convert relative bbox to absolute pixel coordinates
        x = int(bbox_rel["x"] * frame_width)
        y = int(bbox_rel["y"] * frame_height)
        w = int(bbox_rel["w"] * frame_width)
        h = int(bbox_rel["h"] * frame_height)

        # Draw rectangle (top-left and bottom-right corners)
        top_left = (x, y)
        bottom_right = (x + w, y + h)
        color = (255, 255, 255)  # White in BGR
        thickness = 2
        cv2.rectangle(frame, top_left, bottom_right, color, thickness)

        # Save the modified frame
        output_filename = f"frame_{frame_num:05d}.png"
        output_path = os.path.join(output_dir, output_filename)
        cv2.imwrite(output_path, frame)

    # 6. Cleanup
    cap.release()
    print("Done.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Draw TTI bounding boxes on video frames based on JSON data."
    )
    parser.add_argument("--video", required=True, help="Path to the input video file.")
    parser.add_argument(
        "--json",
        required=True,
        help="Path to the filtered JSON file with bounding box data.",
    )
    args = parser.parse_args()

    draw_bounding_boxes(args.video, args.json)
