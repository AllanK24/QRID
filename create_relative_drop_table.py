import json
import pandas as pd
from pathlib import Path
import numpy as np # For handling potential NaN/Inf

# --- Configuration ---
BASE_RESULTS_DIR = Path("/home/omni/Programming/QRID/QRID/results") # Adjust if needed
MODEL_STEMS = ['yolo12n', 'yolo12s', 'yolo12m', 'yolo12l', 'yolo12x'] # Add all models you tested (e.g., 'yolo12m', 'yolo12s')

# Mapping from folder names to display names for degradations
# IMPORTANT: Ensure folder names match EXACTLY what's in your directory structure
# Note: Corrected 'daata_jpeg_heavy' typo based on your example structure
DEGRADATION_MAP = {
    "coco": "Clean",
    "data_blurry_low": "Blurry Low",
    "data_blurry_medium": "Blurry Medium",
    "data_coco_val_mixed_degrad_50pct": "Mixed Degrad. (50%)",
    "data_contrast_low": "Contrast Low", # Check spacing in your actual folder name
    "data_jpeg_heavy": "JPEG Heavy",     # Corrected potential typo based on description
    "data_noisy_low": "Noisy Low",
    "data_noisy_medium": "Noisy Medium"
}
# Get ordered list of validation folder names (excluding clean)
DEGRADATION_FOLDERS_ORDERED = [
    "data_blurry_low",
    "data_blurry_medium",
    "data_contrast_low",
    "data_jpeg_heavy",
    "data_noisy_low",
    "data_noisy_medium",
    "data_coco_val_mixed_degrad_50pct"
]

# --- Data Loading Function ---
def load_json_result(file_path: Path):
    """Loads JSON data from a file."""
    if not file_path.is_file():
        print(f"Warning: Result file not found: {file_path}")
        return None
    try:
        with open(file_path, 'r') as f:
            data = json.load(f)
        return data
    except json.JSONDecodeError:
        print(f"Warning: Error decoding JSON from: {file_path}")
        return None
    except Exception as e:
        print(f"Warning: Could not read file {file_path}: {e}")
        return None

# --- Main Data Processing ---
all_model_results = {} # Store raw mAP results: {model: {config: {degradation_folder: mAP}}}

print("Starting data parsing...")
for model_stem in MODEL_STEMS:
    print(f"Processing model: {model_stem}")
    model_results = {}

    # 1. Process FP32
    fp32_path = BASE_RESULTS_DIR / model_stem / "fp32"
    if fp32_path.is_dir():
        fp32_data = {}
        for folder_name, display_name in DEGRADATION_MAP.items():
            json_file = list(fp32_path.glob(f"{folder_name}/{model_stem}*.json")) # Assuming one json per folder
            if json_file:
                 result_data = load_json_result(json_file[0])
                 if result_data and 'mAP50-95' in result_data:
                     fp32_data[folder_name] = result_data['mAP50-95']
                 else:
                     fp32_data[folder_name] = np.nan # Mark missing data
            else:
                 print(f"Warning: No JSON found for {model_stem}/fp32/{folder_name}")
                 fp32_data[folder_name] = np.nan
        model_results["FP32"] = fp32_data
    else:
        print(f"Warning: FP32 results not found for {model_stem}")

    # 2. Process Static INT8
    int8_path = BASE_RESULTS_DIR / model_stem / "static_int8" # Base INT8 path
    if int8_path.is_dir():
        for calib_folder in ["coco_calib_clean", "coco_calib_mixed"]:
            calib_path = int8_path / calib_folder
            if not calib_path.is_dir():
                print(f"Warning: Calibration results '{calib_folder}' not found for {model_stem}")
                continue

            config_name = f"Static_INT8 ({'Clean' if 'clean' in calib_folder else 'Mixed'} Calib)"
            int8_data = {}
            for folder_name, display_name in DEGRADATION_MAP.items():
                json_file = list(calib_path.glob(f"{folder_name}/{model_stem}*.json")) # Assuming one json per folder
                if json_file:
                    result_data = load_json_result(json_file[0])
                    # Extract mAP from the nested benchmark_result structure
                    mAP = np.nan
                    if result_data and 'benchmark_result' in result_data and isinstance(result_data['benchmark_result'], dict):
                         mAP = result_data['benchmark_result'].get('mAP50-95', np.nan)
                    elif result_data and 'mAP50-95' in result_data: # Fallback if structure was flat before
                         mAP = result_data.get('mAP50-95', np.nan)

                    int8_data[folder_name] = mAP

                else:
                    print(f"Warning: No JSON found for {model_stem}/static_int8/{calib_folder}/{folder_name}")
                    int8_data[folder_name] = np.nan
            model_results[config_name] = int8_data
    else:
        print(f"Warning: Static INT8 results not found for {model_stem}")

    if model_results:
        all_model_results[model_stem] = model_results

print("\nData parsing complete.")

# --- Calculate Relative Drops and Create Tables ---
print("\nGenerating Relative Drop Tables...")

pd.set_option('display.precision', 2) # Set display precision for percentages

for model_stem, configs_data in all_model_results.items():
    print(f"\n--- Results for Model: {model_stem} ---")

    relative_drops = []

    # Get baselines (Clean mAP)
    baselines = {}
    for config_name, data in configs_data.items():
        baselines[config_name] = data.get("coco", np.nan)
        if np.isnan(baselines[config_name]):
             print(f"Warning: Clean baseline mAP missing for {config_name}. Cannot calculate relative drops.")

    # Calculate drops for each degradation
    for folder_name in DEGRADATION_FOLDERS_ORDERED:
        if folder_name == "coco": continue # Skip clean baseline row for drops

        degradation_name = DEGRADATION_MAP.get(folder_name, folder_name) # Get display name
        row = {"Degradation": degradation_name}

        for config_name, data in configs_data.items():
            baseline_map = baselines.get(config_name, np.nan)
            degraded_map = data.get(folder_name, np.nan)
            rel_drop = np.nan # Default to NaN

            if not np.isnan(baseline_map) and not np.isnan(degraded_map):
                if baseline_map > 1e-6: # Avoid division by zero or near-zero
                    rel_drop = ((baseline_map - degraded_map) / baseline_map) * 100.0
                elif degraded_map > 1e-6: # Baseline is zero, degraded is not
                    rel_drop = -np.inf # Indicate infinite relative increase (or large negative drop)
                else: # Both baseline and degraded are zero or near-zero
                     rel_drop = 0.0 # No change from zero

            row[config_name] = rel_drop

        relative_drops.append(row)

    # Create DataFrame
    if not relative_drops:
        print("No data to create table.")
        continue

    df = pd.DataFrame(relative_drops)
    df = df.set_index("Degradation")

    # Reorder columns if necessary (optional)
    column_order = ["FP32", "Static_INT8 (Clean Calib)", "Static_INT8 (Mixed Calib)"]
    # Filter columns to only those present in the DataFrame
    existing_columns = [col for col in column_order if col in df.columns]
    df = df[existing_columns]

    # Format for display
    print("Relative mAP50-95 Drop (%) vs Clean Baseline:")
    # Format as strings with '%' sign, handling NaN
    df_display = df.applymap(lambda x: f"{x:.2f}%" if pd.notna(x) else "N/A")
    print(df_display.to_string())
    print("-" * 80)

    # Optional: Print Markdown table for easy pasting
    # print("\nMarkdown Table:")
    # print(df_display.to_markdown())
    # print("-" * 80)


print("\nAnalysis complete.")