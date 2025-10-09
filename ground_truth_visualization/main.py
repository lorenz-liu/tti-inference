#!/usr/bin/env python3
"""
visualize_annos.py - Visualize Encord annotations on medical videos

Usage:
    python visualize_annos.py -v /path/to/video.mp4 -a /path/to/annotations.json
    python visualize_annos.py -v /path/to/video.mp4 -a /path/to/annotations.json -o /output/dir/
"""

import argparse
import json
import subprocess
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont


def load_encord_annotations(json_path):
    """Load Encord JSON annotation file"""
    with open(json_path, "r") as f:
        return json.load(f)


def get_video_info(video_path):
    """
    Get video properties using ffprobe

    Returns:
    --------
    dict: Contains width, height, fps, total_frames
    """
    cmd = [
        "ffprobe",
        "-v",
        "error",
        "-select_streams",
        "v:0",
        "-count_packets",
        "-show_entries",
        "stream=width,height,r_frame_rate,nb_read_packets",
        "-of",
        "json",
        video_path,
    ]

    result = subprocess.run(cmd, capture_output=True, text=True, check=True)
    info = json.loads(result.stdout)

    stream = info["streams"][0]
    width = int(stream["width"])
    height = int(stream["height"])

    # Parse frame rate
    fps_parts = stream["r_frame_rate"].split("/")
    fps = int(fps_parts[0]) / int(fps_parts[1])

    # Get total frames
    total_frames = int(stream["nb_read_packets"])

    return {"width": width, "height": height, "fps": fps, "total_frames": total_frames}


def read_video_frames(video_path):
    """
    Generator that yields video frames using ffmpeg

    Yields:
    -------
    numpy.ndarray: Frame in RGB format
    """
    # Get video info
    info = get_video_info(video_path)
    width, height = info["width"], info["height"]

    cmd = ["ffmpeg", "-i", video_path, "-f", "rawvideo", "-pix_fmt", "rgb24", "-"]

    process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

    frame_size = width * height * 3  # RGB has 3 channels

    while True:
        raw_frame = process.stdout.read(frame_size)
        if len(raw_frame) != frame_size:
            break

        frame = np.frombuffer(raw_frame, dtype=np.uint8)
        frame = frame.reshape((height, width, 3))

        yield frame

    process.stdout.close()
    process.wait()


def hex_to_rgb(hex_color):
    """Convert hex color to RGB tuple"""
    hex_color = hex_color.lstrip("#")
    return tuple(int(hex_color[i : i + 2], 16) for i in (0, 2, 4))


def draw_bounding_boxes_on_video(
    video_path, annotations_json, output_path, show_labels=True, line_thickness=2
):
    """
    Visualize Encord bounding box annotations on video

    Parameters:
    -----------
    video_path : str
        Path to the original video file
    annotations_json : dict or list
        Loaded JSON annotation data
    output_path : str
        Path to save annotated video
    show_labels : bool
        Whether to show label names on bounding boxes
    line_thickness : int
        Thickness of bounding box lines
    """

    # Get video info
    print("Getting video information...")
    video_info = get_video_info(video_path)
    width = video_info["width"]
    height = video_info["height"]
    fps = video_info["fps"]
    total_frames = video_info["total_frames"]

    print(f"Video properties: {width}x{height} @ {fps:.2f}fps, {total_frames} frames")

    # Process annotations - handle list format from Encord
    if isinstance(annotations_json, list):
        annotations_json = annotations_json[0]  # Take first element if it's a list

    # Extract the labels from the nested structure
    labels_dict = {}
    if "data_units" in annotations_json:
        # Navigate through data_units to find labels
        for data_unit_key, data_unit_value in annotations_json["data_units"].items():
            if "labels" in data_unit_value:
                labels_dict = data_unit_value["labels"]
                break

    if not labels_dict:
        raise ValueError("Could not find labels in the annotation file")

    print(f"Found annotations for {len(labels_dict)} frames")

    # Setup ffmpeg writer
    ffmpeg_cmd = [
        "ffmpeg",
        "-y",  # Overwrite output file
        "-f",
        "rawvideo",
        "-vcodec",
        "rawvideo",
        "-s",
        f"{width}x{height}",
        "-pix_fmt",
        "rgb24",
        "-r",
        str(fps),
        "-i",
        "-",  # Input from pipe
        "-c:v",
        "libx264",
        "-pix_fmt",
        "yuv420p",
        "-preset",
        "medium",
        "-crf",
        "23",
        output_path,
    ]

    ffmpeg_process = subprocess.Popen(
        ffmpeg_cmd, stdin=subprocess.PIPE, stderr=subprocess.PIPE
    )

    print("Processing video...")

    # Try to load a font for better text rendering
    try:
        font = ImageFont.truetype("~/Library/Fonts/Aleo-Thin.ttf", 16)
    except:
        try:
            font = ImageFont.truetype("arial.ttf", 16)
        except:
            font = ImageFont.load_default()

    # Process frames
    frame_idx = 0

    for frame in read_video_frames(video_path):
        # Convert numpy array to PIL Image for drawing
        img = Image.fromarray(frame)
        draw = ImageDraw.Draw(img)

        # Check if current frame has annotations
        frame_key = str(frame_idx)

        if frame_key in labels_dict:
            frame_labels = labels_dict[frame_key]

            # Process objects in this frame
            if "objects" in frame_labels:
                for obj in frame_labels["objects"]:
                    if "boundingBox" in obj:
                        bbox = obj["boundingBox"]

                        # Encord uses normalized coordinates (0-1)
                        # Convert to pixel coordinates
                        x = int(bbox["x"] * width)
                        y = int(bbox["y"] * height)
                        w = int(bbox["w"] * width)
                        h = int(bbox["h"] * height)

                        # Get color
                        color_hex = obj.get("color", "#FF0000")
                        color_rgb = hex_to_rgb(color_hex)

                        # Draw bounding box
                        for i in range(line_thickness):
                            draw.rectangle(
                                [x - i, y - i, x + w + i, y + h + i],
                                outline=color_rgb,
                                width=1,
                            )

                        # Add label if requested
                        if show_labels and "name" in obj:
                            label_text = obj["name"]
                            # Add confidence if available
                            if "confidence" in obj:
                                conf_value = float(obj["confidence"])
                                if conf_value < 1.0:  # Only show if not 1.0
                                    label_text += f" ({conf_value:.2f})"

                            # Get text bounding box
                            bbox_text = draw.textbbox((0, 0), label_text, font=font)
                            text_width = bbox_text[2] - bbox_text[0]
                            text_height = bbox_text[3] - bbox_text[1]

                            # Position label above box, or below if too close to top
                            if y > text_height + 10:
                                label_y = y - text_height - 5
                            else:
                                label_y = y + h + 5

                            # Draw label background
                            draw.rectangle(
                                [
                                    x,
                                    label_y - 2,
                                    x + text_width + 4,
                                    label_y + text_height + 2,
                                ],
                                fill=color_rgb,
                            )

                            # Draw label text
                            draw.text(
                                (x + 2, label_y),
                                label_text,
                                fill=(255, 255, 255),
                                font=font,
                            )

        # Convert back to numpy array and write to ffmpeg
        frame_annotated = np.array(img)
        ffmpeg_process.stdin.write(frame_annotated.tobytes())

        frame_idx += 1

        # Progress indicator
        if frame_idx % 30 == 0 or frame_idx == total_frames:
            progress = (frame_idx / total_frames) * 100
            print(f"Progress: {frame_idx}/{total_frames} frames ({progress:.1f}%)")

    # Close ffmpeg process
    ffmpeg_process.stdin.close()
    ffmpeg_process.wait()

    print(f"\nDone! Annotated video saved to: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Visualize Encord annotations on medical videos",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python visualize_annos.py -v video.mp4 -a annotations.json
  python visualize_annos.py -v video.mp4 -a annotations.json -o /output/dir/
  python visualize_annos.py -v video.mp4 -a annotations.json -o output.mp4

Requirements:
  - ffmpeg must be installed and available in PATH
  - Python packages: numpy, pillow
        """,
    )

    parser.add_argument(
        "-v", "--video", required=True, help="Path to the input video file"
    )

    parser.add_argument(
        "-a",
        "--annotation",
        required=True,
        help="Path to the Encord annotation JSON file",
    )

    parser.add_argument(
        "-o",
        "--output",
        default="./",
        help="Output path (directory or file). Default: current directory (./)",
    )

    parser.add_argument(
        "--no-labels", action="store_true", help="Hide label text on bounding boxes"
    )

    parser.add_argument(
        "--thickness",
        type=int,
        default=2,
        help="Bounding box line thickness. Default: 2",
    )

    args = parser.parse_args()

    # Validate input files
    video_path = Path(args.video)
    annotation_path = Path(args.annotation)

    if not video_path.exists():
        parser.error(f"Video file not found: {args.video}")

    if not annotation_path.exists():
        parser.error(f"Annotation file not found: {args.annotation}")

    # Determine output path
    output_path = Path(args.output)

    # If output is a directory, create filename from input video
    if output_path.is_dir() or str(output_path).endswith("/"):
        output_path.mkdir(parents=True, exist_ok=True)
        output_filename = video_path.stem + "_annotated" + video_path.suffix
        output_path = output_path / output_filename
    else:
        # Create parent directory if it doesn't exist
        output_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"Input video: {video_path}")
    print(f"Annotations: {annotation_path}")
    print(f"Output: {output_path}")
    print("-" * 60)

    # Load annotations
    print("Loading annotations...")
    annotations = load_encord_annotations(str(annotation_path))

    # Process video
    draw_bounding_boxes_on_video(
        video_path=str(video_path),
        annotations_json=annotations,
        output_path=str(output_path),
        show_labels=not args.no_labels,
        line_thickness=args.thickness,
    )


if __name__ == "__main__":
    main()
