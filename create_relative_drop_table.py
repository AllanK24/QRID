import json
import pandas as pd
from pathlib import Path
import numpy as np # For handling potential NaN/Inf

def get_metrics_from_result(result_data):
    """
    Safely extracts mAP50-95 and mAP50,
    handling both nested and flat JSON structures.
    """
    metrics = {'mAP50-95': np.nan, 'mAP50': np.nan} # Default to NaN
    if not result_data:
        return metrics

    data_source = None
    # 1. Check for NESTED structure
    if 'benchmark_result' in result_data and isinstance(result_data['benchmark_result'], dict):
        data_source = result_data['benchmark_result']
    # 2. Fallback to check for FLAT structure
    elif 'mAP50-95' in result_data or 'mAP50' in result_data: # Check if either key exists
        data_source = result_data

    if data_source:
        metrics['mAP50-95'] = data_source.get('mAP50-95', np.nan)
        metrics['mAP50'] = data_source.get('mAP50', np.nan)

    return metrics

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
# --- Main Data Processing ---
all_model_results = {} # Store raw metrics: {model: {config: {degradation_folder: {'mAP50-95': val, 'mAP50': val}}}}

print("Starting data parsing...")
for model_stem in MODEL_STEMS:
    print(f"Processing model: {model_stem}")
    model_results = {} # Results for this specific model

    # --- Process FP32, FP16, Dynamic INT8 ---
    SIMPLE_CONFIGS = {
        "FP32": "fp32",
        "FP16": "fp16",
        "Dynamic INT8": "dynamic_QUInt8",
        }

    for config_name, dir_name in SIMPLE_CONFIGS.items():
        config_path = BASE_RESULTS_DIR / model_stem / dir_name
        if config_path.is_dir():
            config_data = {} # {degradation_folder: {'mAP50-95': val, 'mAP50': val}}
            for folder_name, display_name in DEGRADATION_MAP.items():
                json_files = list(config_path.glob(f"{folder_name}/*.json"))
                if json_files:
                    if len(json_files) > 1:
                         print(f"Warning: Found {len(json_files)} JSONs for {config_path}/{folder_name}. Using first: {json_files[0]}")
                    result_data = load_json_result(json_files[0])
                    # Store the dictionary of metrics
                    config_data[folder_name] = get_metrics_from_result(result_data)
                else:
                    # Store default NaN dict if file not found
                    config_data[folder_name] = {'mAP50-95': np.nan, 'mAP50': np.nan}
            model_results[config_name] = config_data
        else:
             print(f"Warning: {config_name} results directory not found for {model_stem} at {config_path}")

    # --- Process Static INT8 ---
    static_base_path = BASE_RESULTS_DIR / model_stem
    static_dirs = list(static_base_path.glob("static_*"))

    if static_dirs:
       for static_dir_path in static_dirs:
          for calib_folder in ["coco_calib_clean", "coco_calib_mixed"]:
              calib_path = static_dir_path / calib_folder
              if not calib_path.is_dir():
                  continue

              config_name = f"Static INT8 ({'Clean' if 'clean' in calib_folder else 'Mixed'} Calib)"
              int8_data = {} # {degradation_folder: {'mAP50-95': val, 'mAP50': val}}
              for folder_name, display_name in DEGRADATION_MAP.items():
                  json_files = list(calib_path.glob(f"{folder_name}/*.json"))
                  if json_files:
                      result_data = load_json_result(json_files[0])
                      int8_data[folder_name] = get_metrics_from_result(result_data)
                  else:
                       int8_data[folder_name] = {'mAP50-95': np.nan, 'mAP50': np.nan}
              model_results[config_name] = int8_data
    else:
        print(f"Warning: No 'static_*' results directory found for {model_stem}")

    if model_results:
        all_model_results[model_stem] = model_results

print("\nData parsing complete.")

# --- Calculate Relative Drops and Create Tables ---
def generate_relative_drop_table(
    model_stem: str,
    configs_data: dict,
    metric_key: str, # 'mAP50-95' or 'mAP50'
    output_base_dir: Path
):
    """
    Generates and saves a relative drop table for a specific metric.
    """
    print(f"\n--- Generating Table for Metric: {metric_key} (Model: {model_stem}) ---")

    relative_drops = []
    baselines = {}

    # Get baselines for the specific metric
    for config_name, data in configs_data.items():
        # data structure: {degradation_folder: {'mAP50-95': val, 'mAP50': val}}
        clean_metrics = data.get("coco", {}) # Get metrics dict for 'coco'
        baselines[config_name] = clean_metrics.get(metric_key, np.nan) # Get specific metric
        if np.isnan(baselines[config_name]):
             print(f"Warning: Clean baseline {metric_key} missing for {config_name}.")

    # Calculate drops for each degradation
    for folder_name in DEGRADATION_FOLDERS_ORDERED:
        degradation_name = DEGRADATION_MAP.get(folder_name, folder_name)
        row = {"Degradation": degradation_name}

        for config_name in COLUMN_ORDER: # Use defined column order
             if config_name not in configs_data:
                 row[config_name] = np.nan
                 continue

             data = configs_data[config_name] # {degradation_folder: {metric_dict}}
             baseline_map = baselines.get(config_name, np.nan)
             degraded_metrics = data.get(folder_name, {}) # Get metric dict for this folder
             degraded_map = degraded_metrics.get(metric_key, np.nan) # Get specific metric
             rel_drop = np.nan

             if not np.isnan(baseline_map) and not np.isnan(degraded_map):
                if abs(baseline_map) > 1e-7:
                    rel_drop = ((baseline_map - degraded_map) / baseline_map) * 100.0
                elif abs(degraded_map) > 1e-7:
                     rel_drop = -np.inf
                else:
                      rel_drop = 0.0
             row[config_name] = rel_drop
        relative_drops.append(row)

    if not relative_drops:
        print(f"No data to create {metric_key} table for {model_stem}.")
        return

    df = pd.DataFrame(relative_drops)
    df = df.set_index("Degradation")

    existing_columns = [col for col in COLUMN_ORDER if col in df.columns]
    df = df[existing_columns]

    # --- Output ---
    # Create subdirectory for the metric
    metric_output_dir = output_base_dir / metric_key.replace('-','_') # mAP50_95 or mAP50
    metric_output_dir.mkdir(parents=True, exist_ok=True)

    # Format for display
    table_title = f"Relative {metric_key} Drop (%) vs Clean Baseline for {model_stem}:"
    print(table_title)

    def format_value(x):
        if pd.isna(x): return "N/A"
        if np.isinf(x): return "Inf"
        return f"{x:.2f}%"

    df_display = df.map(format_value) # Use map instead of applymap

    print(df_display.to_string())

    # Save CSV (raw numbers)
    csv_filename = metric_output_dir / f"{model_stem}_{metric_key}_relative_drops.csv"
    try:
        df.to_csv(csv_filename, index=True, float_format='%.4f') # Save raw floats
        print(f"Saved raw data to {csv_filename}")
    except Exception as e:
        print(f"Error saving CSV {csv_filename}: {e}")

    # Optional: Save formatted display table to txt
    txt_filename = metric_output_dir / f"{model_stem}_{metric_key}_relative_drops_display.txt"
    try:
        with open(txt_filename, 'w') as f:
            f.write(table_title + "\n\n")
            f.write(df_display.to_string())
        print(f"Saved display table to {txt_filename}")
    except Exception as e:
        print(f"Error saving display txt {txt_filename}: {e}")

    print("-" * (len(table_title) if len(table_title) > 80 else 80) )


# --- Call Table Generation Function ---

print("\nGenerating Relative Drop Tables...")
pd.set_option('display.precision', 2) # Set display precision for percentages

# Define where to save the tables
TABLE_OUTPUT_DIR = Path("./relative_drop_tables") # Base directory for tables

for model_stem, configs_data in all_model_results.items():
    model_table_output_dir = TABLE_OUTPUT_DIR / model_stem
    generate_relative_drop_table(model_stem, configs_data, 'mAP50-95', model_table_output_dir)
    generate_relative_drop_table(model_stem, configs_data, 'mAP50', model_table_output_dir)


print("\nAnalysis complete.")