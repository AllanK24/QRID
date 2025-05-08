import json
import pandas as pd
from pathlib import Path
import numpy as np # For handling potential NaN/Inf
from collections import defaultdict

# --- Configuration ---
BASE_RESULTS_DIR = Path("/home/omni/Programming/QRID/QRID/results") # Adjust if needed
MODEL_STEMS = ['yolo12n', 'yolo12s', 'yolo12m', 'yolo12l', 'yolo12x'] # Add all models you tested

# Mapping from folder names to display names for degradations
# Uses the FOLDER names found inside the precision/calibration dirs
DEGRADATION_MAP = {
    "coco": "Clean",
    "data_blurry_low": "Blurry Low",
    "data_blurry_medium": "Blurry Medium",
    "data_coco_val_mixed_degrad_50pct": "Mixed Degrad. (50%)",
    "data_contrast_low": "Contrast Low",
    "data_jpeg_heavy": "JPEG Heavy",
    "data_noisy_low": "Noisy Low",
    "data_noisy_medium": "Noisy Medium"
}

# Defines the order for rows in the final table
DEGRADATION_FOLDERS_ORDERED = [
     "data_blurry_low",
     "data_blurry_medium",
     "data_contrast_low",
     "data_jpeg_heavy",
     "data_noisy_low",
     "data_noisy_medium",
     "data_coco_val_mixed_degrad_50pct"
 ]

# Defines the order for columns in the final table
COLUMN_ORDER = [
             "FP32",
             "FP16",
             "Dynamic INT8",
             "Static INT8 (Clean Calib)",
             "Static INT8 (Mixed Calib)"
             ]

# --- Helper Functions ---

def load_json_result(file_path: Path):
    """Loads JSON data from a file."""
    if not file_path.is_file():
        # print(f"Debug: Result file not found: {file_path}") # Keep for debugging if needed
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

def get_map_from_result(result_data):
    """
    Safely extracts mAP50-95, handling both nested and flat JSON structures.
    """
    mAP = np.nan # Default to Not a Number
    if not result_data:
        return mAP

    # 1. Check for NESTED structure first
    if 'benchmark_result' in result_data and isinstance(result_data['benchmark_result'], dict):
         mAP = result_data['benchmark_result'].get('mAP50-95', np.nan)
    # 2. Fallback to check for FLAT structure
    elif 'mAP50-95' in result_data:
         mAP = result_data.get('mAP50-95', np.nan)

    return mAP

# --- Main Data Processing ---
all_model_results = {} # Store raw mAP results: {model: {config: {degradation_folder: mAP}}}

print("Starting data parsing...")
for model_stem in MODEL_STEMS:
    print(f"Processing model: {model_stem}")
    model_results = {} # Results for this specific model

    # --- Process FP32, FP16, Dynamic INT8 ---
    # Map Display Name to Directory Name for these simpler cases
    SIMPLE_CONFIGS = {
        "FP32": "fp32",
        "FP16": "fp16",
        "Dynamic INT8": "dynamic_QUInt8", # Using QUINT8 based on your JSON example
        }

    for config_name, dir_name in SIMPLE_CONFIGS.items():
        config_path = BASE_RESULTS_DIR / model_stem / dir_name
        if config_path.is_dir():
            config_data = {}
            for folder_name, display_name in DEGRADATION_MAP.items():
                # Assume one json file inside the degradation folder
                json_files = list(config_path.glob(f"{folder_name}/*.json"))
                if json_files:
                    if len(json_files) > 1:
                         print(f"Warning: Found {len(json_files)} JSONs for {config_path}/{folder_name}. Using first: {json_files[0]}")
                    result_data = load_json_result(json_files[0])
                    config_data[folder_name] = get_map_from_result(result_data)
                else:
                    # config_data[folder_name] = np.nan # Mark missing data
                    pass # Only add key if data found, or default to NaN later
            model_results[config_name] = config_data
        else:
             print(f"Warning: {config_name} results directory not found for {model_stem} at {config_path}")


    # --- Process Static INT8 (Has extra calibration folder layer) ---
    static_base_path = BASE_RESULTS_DIR / model_stem # Base path for this model
    # Assuming the static results live under a subfolder reflecting the type, e.g. static_QInt8_QInt8
    # If they are directly under /static_int8/, adjust the path structure here.
    # We will glob to find directories starting with "static_" to be more robust
    static_dirs = list(static_base_path.glob("static_*"))

    if static_dirs:
      # If you only have one static dir, e.g. "static_QInt8_QInt8" use static_dirs[0]
      # If you might have others, you may need to loop or be more specific
       for static_dir_path in static_dirs: # Handle potentially multiple static_TYPE_TYPE dirs
          for calib_folder in ["coco_calib_clean", "coco_calib_mixed"]:
              calib_path = static_dir_path / calib_folder
              if not calib_path.is_dir():
                  # print(f"Debug: Calibration path '{calib_path}' not found.")
                  continue # Silently skip if this specific calibration wasn't run

              config_name = f"Static INT8 ({'Clean' if 'clean' in calib_folder else 'Mixed'} Calib)"
              int8_data = {}
              for folder_name, display_name in DEGRADATION_MAP.items():
                  json_files = list(calib_path.glob(f"{folder_name}/*.json"))
                  if json_files:
                      result_data = load_json_result(json_files[0])
                      int8_data[folder_name] = get_map_from_result(result_data)
                  else:
                       # int8_data[folder_name] = np.nan # Mark missing
                       pass # Only add key if data found

              model_results[config_name] = int8_data
    else:
        print(f"Warning: No 'static_*' results directory found for {model_stem}")

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
       # if folder_name == "coco": continue # Skip clean baseline row for drops table

        degradation_name = DEGRADATION_MAP.get(folder_name, folder_name) # Get display name
        row = {"Degradation": degradation_name}

        for config_name in COLUMN_ORDER: # Use defined column order
             if config_name not in configs_data:
                 row[config_name] = np.nan # Add NaN if this config doesn't exist for this model
                 continue

             data = configs_data[config_name]
             baseline_map = baselines.get(config_name, np.nan)
             degraded_map = data.get(folder_name, np.nan)
             rel_drop = np.nan # Default to NaN

             if not np.isnan(baseline_map) and not np.isnan(degraded_map):
                if abs(baseline_map) > 1e-7: # Avoid division by zero or near-zero
                    rel_drop = ((baseline_map - degraded_map) / baseline_map) * 100.0
                elif abs(degraded_map) > 1e-7: # Baseline is zero, degraded is not
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

    # Reorder columns to desired, standard order, only keeping those that exist
    existing_columns = [col for col in COLUMN_ORDER if col in df.columns]
    df = df[existing_columns]

    # Format for display
    print("Relative mAP50-95 Drop (%) vs Clean Baseline:")
    # Format as strings with '%' sign, handling NaN and Inf
    def format_value(x):
        if pd.isna(x):
            return "N/A"
        if np.isinf(x):
             return "Inf"
        return f"{x:.2f}%"

    df_display = df.applymap(format_value)

    print(df_display.to_string())
    # df.to_csv(f"{model_stem}_relative_drops.csv", index=True) # Save raw numbers to CSV
    print("-" * 80)

print("\nAnalysis complete.")