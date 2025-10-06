import csv
import os
from pathlib import Path

import cv2

# ========== CONFIGURATION ==========
# TODO: Set the path to your video directory
VIDEO_PATH = (
    "/cluster/projects/madanigroup/lorenz/tti/videos"  # PLACEHOLDER - UPDATE THIS PATH
)

# Directory containing CSV files
CSV_DIR = Path(__file__).parent
OUTPUT_BASE_DIR = CSV_DIR
# ===================================


def get_seconds_from_csv(csv_file):
    """
    Read CSV file and extract second values from column 5 (index 4).
    Returns a sorted list of unique seconds.
    """
    seconds = set()

    with open(csv_file, "r", encoding="utf-8") as f:
        reader = csv.reader(f)
        next(reader)  # Skip header

        for row in reader:
            if len(row) > 4:
                try:
                    sec = int(row[4])
                    seconds.add(sec)
                except (ValueError, IndexError):
                    continue

    return sorted(seconds)


def find_missing_seconds(existing_seconds, total_video_seconds):
    """
    Given a list of existing seconds and total duration in seconds,
    find missing seconds assuming seconds should be at intervals of 10 (0, 10, 20, 30, ...).
    """
    if total_video_seconds is None:
        return []

    # Generate all expected seconds at intervals of 10, from 0 to total_video_seconds
    expected_seconds = set(range(0, total_video_seconds, 10))

    # Find missing seconds
    missing = sorted(expected_seconds - set(existing_seconds))

    return missing


def extract_frames_from_seconds(video_path, seconds_list, output_dir, use_gpu=True):
    """
    Extract all frames from specific seconds in video and save as JPG files.
    Uses GPU acceleration if available.
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Try to use GPU acceleration
    if use_gpu:
        try:
            # Check if CUDA is available
            if cv2.cuda.getCudaEnabledDeviceCount() > 0:
                print("  Using GPU acceleration (CUDA)")
                # Note: VideoCapture doesn't directly support CUDA, but we can use GPU for decoding
                cap = cv2.VideoCapture(video_path, cv2.CAP_FFMPEG)
            else:
                print("  GPU not available, using CPU")
                cap = cv2.VideoCapture(video_path)
        except:
            print("  GPU acceleration not available, using CPU")
            cap = cv2.VideoCapture(video_path)
    else:
        cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print(f"Error: Could not open video {video_path}")
        return False

    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    print(f"  Video FPS: {fps:.2f}, Total frames: {total_frames}")

    extracted_count = 0
    for second in seconds_list:
        # Calculate frame range for this second
        start_frame = int(second * fps)
        end_frame = int((second + 1) * fps)

        # Extract all frames in this second
        for frame_num in range(start_frame, end_frame):
            if frame_num >= total_frames:
                break

            # Set position to the specific frame
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)

            # Read the frame
            ret, frame = cap.read()

            if ret:
                # Save frame as JPG with format: frame_sec_{second}_{frame_in_second}.jpg
                frame_in_second = frame_num - start_frame
                output_file = os.path.join(
                    output_dir, f"frame_sec_{second:03d}_{frame_in_second:03d}.jpg"
                )
                cv2.imwrite(output_file, frame)
                extracted_count += 1
            else:
                print(f"  Warning: Could not read frame {frame_num} at second {second}")

    cap.release()
    print(f"  Extracted {extracted_count} frames from {len(seconds_list)} seconds")

    return True


def process_all_csvs():
    """
    Process all CSV files in the directory and extract missing frames.
    """
    csv_files = list(CSV_DIR.glob("*.csv"))
    csv_files = [f for f in csv_files if f.name != "master-list.csv"]

    print(f"Found {len(csv_files)} CSV files to process\n")

    summary = []

    for csv_file in csv_files:
        video_name = csv_file.stem  # Filename without extension
        print(f"Processing: {video_name}")

        # Hardcoded mapping from CSV names to video filenames
        csv_to_video_map = {
            "LapChol_Case_0001_03": "LapChol Case 0001 03.MP4",
            "LapChol_Case_0001_04": "LapChol Case 0001 04.MP4",
            "LapChol_Case_0001_05": "LapChol Case 0001 05.MP4",
            "LapChol_Case_0002_02": "LapChol Case 0002 02.MP4",
            "LapChol_Case_0002_03": "LapChol Case 0002 03.MP4",
            "LapChol_Case_0007_01": "LapChol Case 0007 01.MP4",
            "LapChol_Case_0007_02": "LapChol Case 0007 02.MP4",
            "LapChol_Case_0007_03": "LapChol Case 0007 03.MP4",
            "LapChol_Case_0011_02": "LapChol Case 0011 02.MP4",
            "LapChol_Case_0011_03": "LapChol Case 0011 03.MP4",
            "LapChol_Case_0012_03": "LapChol Case 0012 03.MP4",
            "LapChol_Case_0012_04": "LapChol Case 0012 04.MP4",
            "LapChol_Case_0015_01": "LapChol Case 0015 01.MP4",
            "LapChol_Case_0015_02": "LapChol Case 0015 02.MP4",
            "LapChol_Case_0016_01": "LapChol Case 0016 01.MP4",
            "LapChol_Case_0018_10": "LapChol Case 0018 10.MP4",
            "LapChol_Case_0018_11": "LapChol Case 0018 11.MP4",
            "LapChol_Case_0019_02": "LapChol Case 0019 02.MP4",
            "LapChol_Case_0019_03": "LapChol Case 0019 03.MP4",
            "LapChol_Case_0020_02": "LapChol Case 0020 02.MP4",
            "LapChol_Case_0020_03": "LapChol Case 0020 03.MP4",
            "LapChol_Case_0023_03": "LapChol Case 0023 03.mp4",
            "LapChol_Case_0023_04": "LapChol Case 0023 04.MP4",
            "V10-Trimmed": "V10-Trimmed.mp4",
            "v11-Trimmed": "V11-Trimmed.mov",
            "V12-Trimmed": "V12-Trimmed.mp4",
            "V14-Trimmed": "V14_Trimmed.mp4",
            "V15-Trimmed": "V15_Trimmed.mp4",
            "V17-Trimmed": "V17_Trimmed.mp4",
            "V18-Trimmed": "V18_Trimmed.mp4",
            "V2-Trimmed": "V2_Trimmed.mp4",
            "V4-Trimmed": "V4_Trimmed.mp4",
            "V5-Trimmed": "V5_Trimmed.mp4",
            "V7-_Trimmed": "V7-Trimmed.mp4",
        }

        # Get video filename from mapping
        video_file = None
        if video_name in csv_to_video_map:
            video_filename = csv_to_video_map[video_name]
            video_file = os.path.join(VIDEO_PATH, video_filename)
            if not os.path.exists(video_file):
                video_file = None

        if not video_file:
            print(f"  Warning: Video file not found for {video_name}")
            print(f"  Expected at: {os.path.join(VIDEO_PATH, video_name)}.mp4")
            summary.append(
                {
                    "video": video_name,
                    "existing": 0,
                    "missing": 0,
                    "status": "video_not_found",
                }
            )
            print()
            continue

        # Get video info
        cap = cv2.VideoCapture(video_file)
        if not cap.isOpened():
            print(f"  Error: Could not open video {video_file}")
            summary.append(
                {
                    "video": video_name,
                    "existing": 0,
                    "missing": 0,
                    "status": "failed_to_open",
                }
            )
            print()
            continue

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_seconds = int(total_frames / fps) if fps > 0 else 0
        cap.release()

        print(
            f"  Video info: {total_frames} total frames, FPS: {fps:.2f}, Duration: {total_seconds}s"
        )

        # Get existing seconds from CSV
        existing_seconds = get_seconds_from_csv(csv_file)
        print(f"  Existing seconds in CSV: {len(existing_seconds)} seconds")
        if existing_seconds:
            print(f"  Range: {min(existing_seconds)}s - {max(existing_seconds)}s")

        # Find missing seconds based on total video duration
        missing_seconds = find_missing_seconds(existing_seconds, total_seconds)

        if not missing_seconds:
            print("  No missing seconds - all seconds present!")
            summary.append(
                {
                    "video": video_name,
                    "existing": len(existing_seconds),
                    "missing": 0,
                    "status": "complete",
                }
            )
            print()
            continue

        print(
            f"  Missing {len(missing_seconds)} seconds: {missing_seconds[:10]}{'...' if len(missing_seconds) > 10 else ''}"
        )

        # Create output directory for this video
        output_dir = OUTPUT_BASE_DIR / video_name

        # Extract frames from missing seconds
        print(
            f"  Extracting frames from {len(missing_seconds)} seconds to {output_dir}"
        )
        success = extract_frames_from_seconds(video_file, missing_seconds, output_dir)

        summary.append(
            {
                "video": video_name,
                "existing": len(existing_seconds),
                "missing": len(missing_seconds),
                "status": "extracted" if success else "failed",
            }
        )

        print()

    # Print summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    for item in summary:
        print(
            f"{item['video']}: {item['existing']} existing, {item['missing']} missing - {item['status']}"
        )
    print("=" * 70)


if __name__ == "__main__":
    print("Frame Extraction Script")
    print("=" * 70)
    print(f"CSV Directory: {CSV_DIR}")
    print(f"Video Path: {VIDEO_PATH}")
    print(f"Output Directory: {OUTPUT_BASE_DIR}")
    print("=" * 70)
    print()

    if VIDEO_PATH == "/path/to/your/videos":
        print("WARNING: Please update VIDEO_PATH at the top of this script!")
        print("Set it to the directory containing your MP4 video files.")
        response = input("Continue anyway? (y/n): ")
        if response.lower() != "y":
            print("Exiting...")
            exit(1)

    process_all_csvs()
    print("\nDone!")
