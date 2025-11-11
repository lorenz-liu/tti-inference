import os
import json
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde

def analyze_predictions():
    """
    Analyzes the performance of a ViT model by comparing its predictions
    against ground truth data.
    """
    ground_truth_dir = 'temporal-ground-truths'
    inferences_dir = 'inferences'
    output_json_path = 'analysis_results.json'
    
    all_percentages = []
    detailed_results = []

    ground_truth_files = [f for f in os.listdir(ground_truth_dir) if f.endswith('.json')]
    print(f"Found {len(ground_truth_files)} ground truth files to analyze.")

    for gt_filename in ground_truth_files:
        print(f"\n--- Processing file: {gt_filename} ---")
        gt_filepath = os.path.join(ground_truth_dir, gt_filename)
        
        pred_filename_v1 = gt_filename
        pred_filename_v2 = "pred_" + gt_filename
        
        pred_filepath = os.path.join(inferences_dir, pred_filename_v2)
        if not os.path.exists(pred_filepath):
            pred_filepath = os.path.join(inferences_dir, pred_filename_v1)
            if not os.path.exists(pred_filepath):
                print(f"Warning: Could not find prediction file for {gt_filename}")
                continue

        try:
            with open(gt_filepath, 'r') as f:
                gt_data = json.load(f)
            
            with open(pred_filepath, 'r') as f:
                pred_data = json.load(f)
        except (json.JSONDecodeError, FileNotFoundError) as e:
            print(f"Error reading or parsing file for {gt_filename}: {e}")
            continue

        # --- Process Ground Truth ---
        start_frames = []
        end_frames = []
        
        if not gt_data or 'data_units' not in gt_data[0]:
            print(f"Warning: Ground truth file {gt_filename} has unexpected format.")
            continue

        gt_data_unit = list(gt_data[0]['data_units'].values())[0]

        for frame_str, frame_data in gt_data_unit['labels'].items():
            for obj in frame_data.get('objects', []):
                if obj.get('name') == 'Start of TTI':
                    start_frames.append(int(frame_str))
                elif obj.get('name') == 'End of TTI ':
                    end_frames.append(int(frame_str))
        
        start_frames.sort()
        end_frames.sort()

        interaction_periods = []
        if len(start_frames) != len(end_frames) or not start_frames:
            print(f"Warning: Mismatch or no TTI markers in {gt_filename}. Pairing starts to nearest ends.")
            used_end_frames = set()
            for start in start_frames:
                best_end = -1
                for end in end_frames:
                    if end > start and end not in used_end_frames:
                        if best_end == -1 or end < best_end:
                            best_end = end
                if best_end != -1:
                    interaction_periods.append((start, best_end))
                    used_end_frames.add(best_end)
        else:
            interaction_periods = list(zip(start_frames, end_frames))
        
        print(f"Found {len(interaction_periods)} interaction period(s) in ground truth.")

        # --- Process Predictions ---
        predicted_interaction_frames = set()
        if not pred_data or 'data_units' not in pred_data[0]:
            print(f"Warning: Prediction file for {gt_filename} has unexpected format.")
            continue
            
        pred_data_unit = list(pred_data[0]['data_units'].values())[0]

        for frame_str, frame_data in pred_data_unit['labels'].items():
            is_interaction = False
            for obj in frame_data.get('objects', []):
                if obj.get('name') == 'Start of TTI':
                    is_interaction = True
                    break
            if is_interaction:
                predicted_interaction_frames.add(int(frame_str))

        # --- Calculate Percentages ---
        for i, (start_frame, end_frame) in enumerate(interaction_periods):
            if start_frame > end_frame:
                continue
                
            total_frames_in_period = end_frame - start_frame + 1
            correctly_predicted_frames = 0
            
            for frame in range(start_frame, end_frame + 1):
                if frame in predicted_interaction_frames:
                    correctly_predicted_frames += 1
            
            if total_frames_in_period > 0:
                percentage = (correctly_predicted_frames / total_frames_in_period) * 100
                all_percentages.append(percentage)
                
                result_entry = {
                    "ground_truth_file": gt_filename,
                    "interaction_index": i + 1,
                    "start_frame": start_frame,
                    "end_frame": end_frame,
                    "total_frames_in_period": total_frames_in_period,
                    "correctly_predicted_frames": correctly_predicted_frames,
                    "percentage": round(percentage, 2)
                }
                detailed_results.append(result_entry)
                print(f"  - Interaction {i+1} ({start_frame}-{end_frame}): {correctly_predicted_frames}/{total_frames_in_period} frames correct ({percentage:.2f}%)")

    if not all_percentages:
        print("\nNo valid interaction periods found across all files. Cannot generate plot.")
        return

    # --- Save Detailed Results to JSON ---
    with open(output_json_path, 'w') as f:
        json.dump(detailed_results, f, indent=4)
    print(f"\nDetailed analysis results saved to {output_json_path}")

    # --- Plotting ---
    percentages_array = np.array(all_percentages)
    
    plt.figure(figsize=(12, 7))
    
    try:
        kde = gaussian_kde(percentages_array)
        x_range = np.linspace(0, 100, 500)
        y_range = kde(x_range)
        
        plt.plot(x_range, y_range, color='darkblue', lw=2, label='Density Curve')
        plt.fill_between(x_range, y_range, color='lightblue', alpha=0.4)
    except np.linalg.LinAlgError:
        print("Warning: Could not generate KDE plot (likely all percentages are the same). Falling back to a histogram.")
        plt.hist(percentages_array, bins=20, density=True, color='lightblue', ec='darkblue')

    plt.title('Distribution of Correctly Predicted Frames within Ground Truth Interactions', fontsize=16)
    plt.xlabel('Percentage of Frames Correctly Predicted (%)', fontsize=12)
    plt.ylabel('Density', fontsize=12)
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    plt.xlim(0, 100)
    plt.ylim(bottom=0)
    plt.legend()
    
    output_filename = 'interaction_prediction_distribution.png'
    plt.savefig(output_filename)
    print(f"Analysis complete. Plot saved to {output_filename}")
    
    plt.show(block=False)
    plt.pause(2)
    plt.close()


if __name__ == '__main__':
    # This script assumes you have the necessary libraries installed.
    # You can install them using pip:
    # pip install numpy matplotlib scipy
    analyze_predictions()
