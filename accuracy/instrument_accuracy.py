import json
import argparse
import math
import os


def get_box_center(box):
    """Calculates the center of a bounding box."""
    return (box["x"] + box["w"] / 2, box["y"] + box["h"] / 2)


def distance(p1, p2):
    """Calculates the Euclidean distance between two points."""
    return math.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)


def normalize_instrument_name(name):
    """Normalizes instrument names for comparison (e.g., 'grasper_1' -> 'grasper')."""
    if not name:
        return ""
    return name.split("_")[0]


def main():
    """Main function to compare instrument predictions against ground truth."""
    parser = argparse.ArgumentParser(
        description="Compare instrument predictions with ground truth."
    )
    parser.add_argument(
        "--pred", required=True, help="Path to the prediction JSON file."
    )
    parser.add_argument(
        "--gt", required=True, help="Path to the ground truth JSON file."
    )
    args = parser.parse_args()

    output_filename = f"{os.path.basename(args.gt)[:-5]}_accuracy.txt"
    output_lines = []

    denominator = 0
    nominator = 0

    with open(args.pred, "r") as f:
        predictions_data = json.load(f)

    with open(args.gt, "r") as f:
        ground_truth_data = json.load(f)

    # Assuming a single video's data in the JSON file
    pred_labels = list(predictions_data[0]["data_units"].values())[0]["labels"]
    gt_labels = list(ground_truth_data[0]["data_units"].values())[0]["labels"]
    gt_object_answers = ground_truth_data[0]["object_answers"]

    for frame_num, pred_frame_data in pred_labels.items():
        if frame_num not in gt_labels:
            # print(
            #     f"Warning: Frame {frame_num} found in predictions but not in ground truth."
            # )
            continue

        gt_frame_data = gt_labels[frame_num]
        pred_objects = pred_frame_data.get("objects", [])
        gt_objects = gt_frame_data.get("objects", [])

        # Create a list of ground truth objects with details for matching
        gt_object_details = []
        for gt_obj in gt_objects:
            obj_hash = gt_obj.get("objectHash")
            if obj_hash and obj_hash in gt_object_answers:
                for classification in gt_object_answers[obj_hash].get(
                    "classifications", []
                ):
                    if classification.get("value") == "type_of_instrument":
                        answer = classification.get("answers", [{}])[0]
                        instrument_name = answer.get("value")
                        if instrument_name:
                            gt_object_details.append(
                                {
                                    "box": gt_obj["boundingBox"],
                                    "center": get_box_center(gt_obj["boundingBox"]),
                                    "instrument": instrument_name,
                                }
                            )
                        break

        # A set to track which ground truth objects have been matched
        matched_gt_indices = set()

        for pred_obj in pred_objects:
            predicted_instrument = pred_obj.get("tool_info", {}).get("name")
            if not predicted_instrument or "tti_bounding_box" not in pred_obj:
                continue

            pred_center = get_box_center(pred_obj["tti_bounding_box"])

            # Find the closest, available ground truth object
            min_dist = float("inf")
            closest_gt_idx = -1
            for i, gt_detail in enumerate(gt_object_details):
                if i in matched_gt_indices:
                    continue
                dist = distance(pred_center, gt_detail["center"])
                if dist < min_dist:
                    min_dist = dist
                    closest_gt_idx = i

            if closest_gt_idx != -1:
                # Mark this ground truth object as matched
                matched_gt_indices.add(closest_gt_idx)

                closest_gt = gt_object_details[closest_gt_idx]
                true_instrument = closest_gt["instrument"]

                # Normalize names for a more robust comparison
                normalized_pred = normalize_instrument_name(predicted_instrument)
                normalized_true = normalize_instrument_name(true_instrument)

                if normalized_pred == normalized_true:
                    output_lines.append(
                        f"In frame {frame_num}, True Prediction: {predicted_instrument}"
                    )
                    denominator += 1
                    nominator += 1
                else:
                    output_lines.append(
                        f"In frame {frame_num}, False Prediction: {predicted_instrument} should be {true_instrument}"
                    )
                    denominator += 1
            else:
                # No matching ground truth object found for this prediction
                output_lines.append(
                    f"In frame {frame_num}, False Prediction: {predicted_instrument} should be No Instrument"
                )

    if denominator > 0:
        accuracy = nominator / denominator
        output_lines.append(str(accuracy))
    else:
        output_lines.append("Cannot calculate accuracy, denominator is 0.")

    with open(output_filename, "w") as f:
        f.write("\n".join(output_lines))


if __name__ == "__main__":
    main()
