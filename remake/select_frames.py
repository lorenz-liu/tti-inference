import argparse
import json
import os


def filter_tti_labels_by_value(json_data):
    """
    Filters a list of label data to select frames containing exactly one object
    where the object's value is 'start_of_tti'.
    """
    filtered_data = []
    for item in json_data:
        new_item = item.copy()
        new_data_units = {}
        for data_hash, data_unit in item.get("data_units", {}).items():
            new_data_unit = data_unit.copy()
            new_labels = {}
            for frame_num, label_data in data_unit.get("labels", {}).items():
                objects = label_data.get("objects", [])

                if len(objects) == 1 and objects[0].get("value") == "start_of_tti":
                    new_labels[frame_num] = label_data

            if new_labels:
                new_data_unit["labels"] = new_labels
                new_data_units[data_hash] = new_data_unit

        if new_data_units:
            new_item["data_units"] = new_data_units
            filtered_data.append(new_item)

    return filtered_data


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Select frames with exactly one object with value 'start_of_tti'."
    )
    parser.add_argument("--json", required=True, help="Path to the input JSON file.")
    args = parser.parse_args()

    try:
        with open(args.json, "r") as f:
            data = json.load(f)
    except FileNotFoundError:
        print(f"Error: Input file not found at {args.json}")
        exit(1)
    except json.JSONDecodeError:
        print(f"Error: Could not decode JSON from {args.json}")
        exit(1)

    filtered_json = filter_tti_labels_by_value(data)

    base, ext = os.path.splitext(args.json)
    output_filename = f"{base}_filtered{ext}"

    with open(output_filename, "w") as f:
        json.dump(filtered_json, f, indent=2)

    print(f"Filtered data written to {output_filename}")
