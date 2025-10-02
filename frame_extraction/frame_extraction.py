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


def get_frames_from_csv(csv_file):
    """
    Read CSV file and extract frame numbers from column 5 (index 4).
    Returns a sorted list of unique frame numbers.
    """
    frames = set()

    with open(csv_file, "r", encoding="utf-8") as f:
        reader = csv.reader(f)
        next(reader)  # Skip header

        for row in reader:
            if len(row) > 4:
                try:
                    frame_num = int(row[4])
                    frames.add(frame_num)
                except (ValueError, IndexError):
                    continue

    return sorted(frames)


def find_missing_frames(existing_frames, total_video_frames):
    """
    Given a list of existing frame numbers and total frames in video,
    find missing frames assuming frames should be at intervals of 10 (0, 10, 20, 30, ...).
    """
    if total_video_frames is None:
        return []

    # Generate all expected frames at intervals of 10, from 0 to total_video_frames
    expected_frames = set(range(0, total_video_frames, 10))

    # Find missing frames
    missing = sorted(expected_frames - set(existing_frames))

    return missing


def extract_frames(video_path, frame_numbers, output_dir, use_gpu=True):
    """
    Extract specific frames from video and save as JPG files.
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
    for frame_num in frame_numbers:
        # Set position to the specific frame
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)

        # Read the frame
        ret, frame = cap.read()

        if ret:
            # Save frame as JPG
            output_file = os.path.join(output_dir, f"frame_{frame_num:06d}.jpg")
            cv2.imwrite(output_file, frame)
            extracted_count += 1
        else:
            print(f"  Warning: Could not read frame {frame_num}")

    cap.release()
    print(f"  Extracted {extracted_count}/{len(frame_numbers)} frames")

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

        # Get total frames from video
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
        cap.release()

        print(f"  Video info: {total_frames} total frames, FPS: {fps:.2f}")

        # Get existing frames from CSV
        existing_frames = get_frames_from_csv(csv_file)
        print(f"  Existing frames in CSV: {len(existing_frames)} frames")
        if existing_frames:
            print(f"  Range: {min(existing_frames)} - {max(existing_frames)}")

        # Find missing frames based on total video frames
        missing_frames = find_missing_frames(existing_frames, total_frames)

        if not missing_frames:
            print("  No missing frames - all frames present!")
            summary.append(
                {
                    "video": video_name,
                    "existing": len(existing_frames),
                    "missing": 0,
                    "status": "complete",
                }
            )
            print()
            continue

        print(
            f"  Missing {len(missing_frames)} frames: {missing_frames[:10]}{'...' if len(missing_frames) > 10 else ''}"
        )

        # Create output directory for this video
        output_dir = OUTPUT_BASE_DIR / video_name

        # Extract missing frames
        print(f"  Extracting {len(missing_frames)} frames to {output_dir}")
        success = extract_frames(video_file, missing_frames, output_dir)

        summary.append(
            {
                "video": video_name,
                "existing": len(existing_frames),
                "missing": len(missing_frames),
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
