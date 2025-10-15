import csv
import os
from multiprocessing import Pool, cpu_count
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

# Parallel processing configuration
MAX_WORKERS = min(4, cpu_count())  # Use up to 4 workers (CPUs/GPUs)
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


def extract_frames_from_seconds(video_path, seconds_list, output_dir):
    """
    Extract one frame at each specified second from video and save as JPG files.
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print(f"Error: Could not open video {video_path}")
        return False

    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    print(f"  Video FPS: {fps:.2f}, Total frames: {total_frames}")

    extracted_count = 0
    for second in seconds_list:
        # Calculate frame number at this second
        frame_num = int(second * fps)

        if frame_num >= total_frames:
            print(f"  Warning: Second {second} is beyond video duration")
            continue

        # Set position to the specific frame
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)

        # Read the frame
        ret, frame = cap.read()

        if ret:
            # Save frame as JPG with format: frame_sec_{second}.jpg
            output_file = os.path.join(output_dir, f"frame_sec_{second:03d}.jpg")
            cv2.imwrite(output_file, frame)
            extracted_count += 1
        else:
            print(f"  Warning: Could not read frame at second {second}")

    cap.release()
    print(f"  Extracted {extracted_count}/{len(seconds_list)} frames")

    return True


def process_single_video(args):
    """
    Process a single video - designed to be called in parallel.
    Returns a summary dict for this video.
    """
    csv_file, worker_id = args
    video_name = csv_file.stem

    print(f"[Worker {worker_id}] Processing: {video_name}")

    # Hardcoded mapping from CSV names to video filenames
    csv_to_video_map = {
        "LapChol_Case_0023_03": "LapChol Case 0023 03.MP4",
    }

    # Get video filename from mapping
    video_file = None
    if video_name in csv_to_video_map:
        video_filename = csv_to_video_map[video_name]
        video_file = os.path.join(VIDEO_PATH, video_filename)
        if not os.path.exists(video_file):
            video_file = None

    if not video_file:
        print(f"[Worker {worker_id}]   Warning: Video file not found for {video_name}")
        return {
            "video": video_name,
            "existing": 0,
            "missing": 0,
            "status": "video_not_found",
        }

    # Get video info
    cap = cv2.VideoCapture(video_file)
    if not cap.isOpened():
        print(f"[Worker {worker_id}]   Error: Could not open video {video_file}")
        return {
            "video": video_name,
            "existing": 0,
            "missing": 0,
            "status": "failed_to_open",
        }

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_seconds = int(total_frames / fps) if fps > 0 else 0
    cap.release()

    print(
        f"[Worker {worker_id}]   Video info: {total_frames} total frames, FPS: {fps:.2f}, Duration: {total_seconds}s"
    )

    # Get existing seconds from CSV
    existing_seconds = get_seconds_from_csv(csv_file)
    print(
        f"[Worker {worker_id}]   Existing seconds in CSV: {len(existing_seconds)} seconds"
    )
    if existing_seconds:
        print(
            f"[Worker {worker_id}]   Range: {min(existing_seconds)}s - {max(existing_seconds)}s"
        )

    # Find missing seconds based on total video duration
    missing_seconds = find_missing_seconds(existing_seconds, total_seconds)

    if not missing_seconds:
        print(f"[Worker {worker_id}]   No missing seconds - all seconds present!")
        return {
            "video": video_name,
            "existing": len(existing_seconds),
            "missing": 0,
            "status": "complete",
        }

    print(
        f"[Worker {worker_id}]   Missing {len(missing_seconds)} seconds: {missing_seconds[:10]}{'...' if len(missing_seconds) > 10 else ''}"
    )

    # Create output directory for this video
    output_dir = OUTPUT_BASE_DIR / video_name

    # Extract frames from missing seconds
    print(
        f"[Worker {worker_id}]   Extracting frames from {len(missing_seconds)} seconds to {output_dir}"
    )

    success = extract_frames_from_seconds(video_file, missing_seconds, output_dir)

    return {
        "video": video_name,
        "existing": len(existing_seconds),
        "missing": len(missing_seconds),
        "status": "extracted" if success else "failed",
    }


def process_all_csvs():
    """
    Process all CSV files in the directory and extract missing frames using parallel processing.
    """
    csv_files = list(CSV_DIR.glob("*.csv"))
    csv_files = [f for f in csv_files if f.name != "master-list.csv"]

    print(f"Found {len(csv_files)} CSV files to process")
    print(f"Using {MAX_WORKERS} parallel workers (CPUs/GPUs)\n")

    summary = []

    # Prepare arguments for parallel processing (csv_file, worker_id)
    process_args = [(csv_file, i % MAX_WORKERS) for i, csv_file in enumerate(csv_files)]

    # Process videos in parallel
    with Pool(processes=MAX_WORKERS) as pool:
        summary = pool.map(process_single_video, process_args)

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

