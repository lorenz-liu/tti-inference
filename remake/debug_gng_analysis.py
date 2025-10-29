# This script requires opencv-python and numpy.
# You can install them using: pip install opencv-python numpy

import argparse
import json
import os
import cv2
import numpy as np


def visualize_frame_analysis(json_path, gng_video_path, frame_to_debug):
    """_summary_

    Args:
        json_path (_type_): _description_
        gng_video_path (_type_): _description_
        frame_to_debug (_type_): _description_
    """
    # Define HSV color ranges for green and red
    LOWER_GREEN_HSV = np.array([40, 100, 50])
    UPPER_GREEN_HSV = np.array([80, 255, 255])
    LOWER_RED_HSV1 = np.array([0, 100, 50])
    UPPER_RED_HSV1 = np.array([10, 255, 255])
    LOWER_RED_HSV2 = np.array([160, 100, 50])
    UPPER_RED_HSV2 = np.array([179, 255, 255])

    TARGET_WIDTH = 1280
    TARGET_HEIGHT = 720

    # 1. Load JSON and find the specific frame's data
    try:
        with open(json_path, "r") as f:
            data = json.load(f)
    except FileNotFoundError:
        print(f"Error: JSON file not found at {json_path}")
        return

    bbox_rel = None
    for item in data:
        for data_unit in item.get("data_units", {}).values():
            if str(frame_to_debug) in data_unit.get("labels", {}):
                label_data = data_unit["labels"][str(frame_to_debug)]
                bbox_rel = label_data["objects"][0].get("tti_bounding_box")
                break
        if bbox_rel:
            break

    if not bbox_rel:
        print(f"Error: Frame {frame_to_debug} not found in JSON file.")
        return

    # 2. Open video and seek to frame
    cap = cv2.VideoCapture(gng_video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video file at {gng_video_path}")
        return

    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_to_debug)
    ret, frame_from_video = cap.read()
    if not ret:
        print(f"Error: Could not read frame {frame_to_debug} from video.")
        cap.release()
        return

    # 3. Perform resizing/padding logic
    gng_native_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    gng_native_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    needs_resize = (
        gng_native_height != TARGET_HEIGHT or gng_native_width != TARGET_WIDTH
    )

    final_gng_frame = None
    if needs_resize:
        aspect_ratio = gng_native_width / gng_native_height
        new_h = TARGET_HEIGHT
        new_w = int(new_h * aspect_ratio)
        resized = cv2.resize(
            frame_from_video, (new_w, new_h), interpolation=cv2.INTER_NEAREST
        )
        canvas = np.zeros((TARGET_HEIGHT, TARGET_WIDTH, 3), dtype=np.uint8)
        x_offset = (TARGET_WIDTH - new_w) // 2
        canvas[:, x_offset : x_offset + new_w] = resized
        final_gng_frame = canvas
    else:
        final_gng_frame = frame_from_video

    # 4. Perform analysis
    x = int(bbox_rel["x"] * TARGET_WIDTH)
    y = int(bbox_rel["y"] * TARGET_HEIGHT)
    w = int(bbox_rel["w"] * TARGET_WIDTH)
    h = int(bbox_rel["h"] * TARGET_HEIGHT)

    bbox_roi = final_gng_frame[y : y + h, x : x + w]

    # Convert ROI to HSV and create masks
    hsv_roi = cv2.cvtColor(bbox_roi, cv2.COLOR_BGR2HSV)
    go_mask = cv2.inRange(hsv_roi, LOWER_GREEN_HSV, UPPER_GREEN_HSV)
    red_mask1 = cv2.inRange(hsv_roi, LOWER_RED_HSV1, UPPER_RED_HSV1)
    red_mask2 = cv2.inRange(hsv_roi, LOWER_RED_HSV2, UPPER_RED_HSV2)
    nogo_mask = cv2.bitwise_or(red_mask1, red_mask2)

    go_pixels = cv2.countNonZero(go_mask)
    nogo_pixels = cv2.countNonZero(nogo_mask)

    print(f"--- Debugging Frame {frame_to_debug} ---")
    print(f"Bounding Box (x,y,w,h): ({x}, {y}, {w}, {h})")
    print(f"ROI Shape: {bbox_roi.shape}")
    print(f"Using HSV ranges for detection.")
    print(f"GO pixels found: {go_pixels}")
    print(f"NOGO pixels found: {nogo_pixels}")

    # 5. Create visualization
    vis_main = final_gng_frame.copy()
    cv2.rectangle(vis_main, (x, y), (x + w, y + h), (0, 255, 255), 2)  # Yellow box

    vis_h, vis_w = TARGET_HEIGHT // 2, TARGET_WIDTH // 2

    vis_main_small = cv2.resize(
        vis_main, (vis_w, vis_h), interpolation=cv2.INTER_NEAREST
    )
    cv2.putText(
        vis_main_small,
        "Full Frame + BBox",
        (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 255, 255),
        2,
    )

    vis_roi = cv2.resize(bbox_roi, (vis_w, vis_h), interpolation=cv2.INTER_NEAREST)
    cv2.putText(vis_roi, "ROI", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)

    vis_go_mask = cv2.cvtColor(go_mask, cv2.COLOR_GRAY2BGR)
    vis_go_mask = cv2.resize(
        vis_go_mask, (vis_w, vis_h), interpolation=cv2.INTER_NEAREST
    )
    cv2.putText(
        vis_go_mask, "GO Mask", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2
    )

    vis_nogo_mask = cv2.cvtColor(nogo_mask, cv2.COLOR_GRAY2BGR)
    vis_nogo_mask = cv2.resize(
        vis_nogo_mask, (vis_w, vis_h), interpolation=cv2.INTER_NEAREST
    )
    cv2.putText(
        vis_nogo_mask,
        "NOGO Mask",
        (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 255, 255),
        2,
    )

    top_row = cv2.hconcat([vis_main_small, vis_roi])
    bottom_row = cv2.hconcat([vis_go_mask, vis_nogo_mask])
    final_vis = cv2.vconcat([top_row, bottom_row])

    output_filename = f"debug_frame_{frame_to_debug}.png"
    cv2.imwrite(output_filename, final_vis)
    print(f"Saved visualization to {output_filename}")

    cap.release()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Debug GNG analysis for a single frame."
    )
    parser.add_argument(
        "--filtered_json", required=True, help="Path to the filtered JSON file."
    )
    parser.add_argument(
        "--gng_video", required=True, help="Path to the GO/NOGO video file."
    )
    parser.add_argument(
        "--frame_number",
        required=True,
        type=int,
        help="The specific frame number to debug.",
    )
    args = parser.parse_args()

    visualize_frame_analysis(args.filtered_json, args.gng_video, args.frame_number)
