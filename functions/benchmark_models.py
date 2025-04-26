import os
import json
import shutil
from pathlib import Path
from ultralytics import YOLO
from functions.benchmark import benchmark
from functions.preprocess import preprocess_model
from functions.export_to_onnx import export_to_onnx
from functions.dynamic_quant import dynamic_quantization
from functions.get_device_name import get_cpu_name, get_gpu_name
from functions.static_quant import static_quantization, YOLOCalibrationDataReader, ImageCalibrationDataReader

def benchmark_yolo_fp(
    model_pt_path_str: str, # Changed name for clarity: Path to the original .pt file
    models_base_dir: str, # Base directory to store pt/onnx models (e.g., 'models/')
    results_base_dir: str, # Base directory to store results (e.g., 'results/')
    half: bool=False, # Use half precision (FP16) if True
    onnx_export_kwargs: dict={},
    onnx_preprocess_kwargs: dict={},
    onnx_benchmark_kwargs: dict={}, # Must contain the 'data' key for the dataset yaml
):
    """
    Benchmarks a YOLO FP32/FP16 model by exporting to ONNX, pre-processing,
    evaluating the processed ONNX model, and saving results.

    Args:
        model_pt_path_str (str): Path to the original YOLOv8 PyTorch (.pt) model file.
        models_base_dir (str): Base directory to store organized model files (pt and onnx).
        results_base_dir (str): Base directory to store organized benchmark results.
        onnx_export_kwargs (dict): Arguments for the ONNX export function.
        onnx_preprocess_kwargs (dict): Arguments for the ONNX pre-processing function.
        onnx_benchmark_kwargs (dict): Arguments for the benchmark function (MUST include 'data' key).
    
    Returns:
        dict: A dictionary containing benchmark results including mAP50, mAP50-95, latency, and FPS.
    """
    # --- Ensure Consistency for 'half' ---
    onnx_export_kwargs['half'] = half
    onnx_benchmark_kwargs['half'] = half # Ensure benchmark also knows the target precision
    precision = "fp16" if half else "fp32"
    
    original_pt_path = Path(model_pt_path_str)
    if not original_pt_path.exists():
        print(f"Input PyTorch model not found at {original_pt_path}, downloading model...")
        # Attempt downloading the model automatically if it doesn't exist
        model = YOLO(model_pt_path_str, task="detect")
        original_pt_path = Path(model.ckpt_path)
        print(f"Downloaded model to {original_pt_path}")

    model_stem = original_pt_path.stem # e.g., "yolov8n"

    # --- Define Directory Structure ---
    pt_model_dir = Path(models_base_dir) / "pt" / model_stem / precision
    onnx_model_dir = Path(models_base_dir) / "onnx" / model_stem / precision
    results_dir = Path(results_base_dir) / model_stem / precision

    # --- Create Directories ---
    os.makedirs(pt_model_dir, exist_ok=True)
    os.makedirs(onnx_model_dir, exist_ok=True)
    # Results dir will be created later based on dataset

    # Move the original PT model to the organized directory
    organized_pt_path = pt_model_dir / original_pt_path.name
    try:
        shutil.move(original_pt_path, organized_pt_path)
        print(f"Moved {original_pt_path} to {organized_pt_path}")
    except Exception as e:
        print(f"Error moving model {original_pt_path} to {organized_pt_path}: {e}")
        return None

    # --- Export the Copied PT model to ONNX ---
    exported_onnx_path = pt_model_dir / f"{model_stem}.onnx" # Save initial ONNX next to its PT source
    print(f"Exporting {organized_pt_path} to {exported_onnx_path}...")
    exported_onnx_path = export_to_onnx(
        model_path=organized_pt_path,
        export_kwargs=onnx_export_kwargs,
    )
    
    # Move the exported ONNX model to the ONNX directory
    try:
        shutil.move(exported_onnx_path, onnx_model_dir / str(exported_onnx_path).split("/")[-1])
        print(f"Moved exported ONNX model to {onnx_model_dir}")
    except Exception as e:
        print(f"Error moving exported ONNX model to {onnx_model_dir}: {e}")
        return None

    # --- Preprocess the Exported ONNX model ---
    processed_onnx_path = onnx_model_dir / f"{model_stem}_processed.onnx" # Save processed model here
    try:
        # Assuming preprocess_model takes input and output paths
        preprocess_model(
            input_model_path=onnx_model_dir/f"{model_stem}.onnx",
            output_model_path=processed_onnx_path,
            preprocess_kwargs=onnx_preprocess_kwargs,
        )
        # Check if the output file was actually created
        if not processed_onnx_path.exists():
             raise FileNotFoundError("Preprocessed ONNX file was not created.")
        print("Preprocessed ONNX file was created.")
    except Exception as e:
        print(f"Error preprocessing ONNX model {exported_onnx_path}: {e}")
        return None

    # --- Perform Benchmark on the Preprocessed ONNX model ---
    dataset_yaml_path_str = onnx_benchmark_kwargs.get("data", "coco.yaml")
    dataset_stem = Path(dataset_yaml_path_str).stem # e.g., "coco" or "coco_noisy_low"

    try:
        # Load the processed ONNX model for benchmarking
        model_onnx = YOLO(str(processed_onnx_path)) # Must use string path for YOLO constructor
        benchmark_results_obj = benchmark( # Assuming benchmark returns the results object
            model=model_onnx,
            kwargs=onnx_benchmark_kwargs, # Pass all benchmark args (includes data)
        )
        if benchmark_results_obj is None:
            raise RuntimeError("Benchmark function returned None or failed internally.")
    except Exception as e:
        print(f"Error benchmarking model {processed_onnx_path}: {e}")
        return None

    # --- Extract and Save Results ---
    results_data = {
        "model_name": original_pt_path.stem,
        "model_type": f"{precision.upper()}_ONNX_Processed",
        "dataset": dataset_yaml_path_str,
        "hardware": get_gpu_name() if onnx_benchmark_kwargs.get("device", "cpu") == "cuda" else get_cpu_name(),
        "mAP50-95": benchmark_results_obj.results_dict.get('metrics/mAP50-95(B)', None),
        "mAP50": benchmark_results_obj.results_dict.get('metrics/mAP50(B)', None),
        "latency_ms": benchmark_results_obj.speed.get('inference', None),
        "fps": None,
    }
    if results_data["latency_ms"] is not None:
        results_data["fps"] = 1000.0 / results_data["latency_ms"]

    # Define results file path
    dataset_results_dir = results_dir / dataset_stem
    os.makedirs(dataset_results_dir, exist_ok=True)
    results_json_path = dataset_results_dir / f"{model_stem}_{precision}_benchmark.json"

    print(f"Saving benchmark results to {results_json_path}...")
    try:
        with open(results_json_path, "w") as f:
            json.dump(results_data, f, indent=4) # Use json.dump for proper formatting
        print("Benchmark results saved successfully.")
        return results_data # Return the collected data
    except Exception as e:
        print(f"Error saving results to {results_json_path}: {e}")
        return None # Indicate failure

def benchmark_yolo_dynamic_quant(
    model_pt_path_str: str, # Changed name for clarity: Path to the original .pt file
    models_base_dir: str, # Base directory to store pt/onnx models (e.g., 'models/')
    results_base_dir: str, # Base directory to store results (e.g., 'results/')
    onnx_export_kwargs: dict={}, # Arguments for the ONNX export function
    onnx_preprocess_kwargs: dict={}, # Arguments for the ONNX pre-processing function
    onnx_dynamic_quant_kwargs: dict={}, # Arguments for dynamic quantization
    onnx_benchmark_kwargs: dict={}, # Must contain the 'data' key for the dataset yaml
):
    """
    Benchmarks a YOLO model with Dynamic Quantization by exporting to ONNX, pre-processing,
    evaluating the processed ONNX model, and saving results.

    Args:
        model_pt_path_str (str): Path to the original YOLOv8 PyTorch (.pt) model file.
        models_base_dir (str): Base directory to store organized model files (pt and onnx).
        results_base_dir (str): Base directory to store organized benchmark results.
        onnx_export_kwargs (dict): Arguments for the ONNX export function.
        onnx_preprocess_kwargs (dict): Arguments for the ONNX pre-processing function.
        onnx_dynamic_quant_kwargs (dict): Arguments for dynamic quantization.
        onnx_benchmark_kwargs (dict): Arguments for the benchmark function (MUST include 'data' key).
    
    Returns:
        dict: A dictionary containing benchmark results including mAP50, mAP50-95, latency, and FPS.
    """
    # --- Ensure Consistency for 'half', should be False for dynamic quantization ---
    onnx_export_kwargs['half'] = False
    onnx_benchmark_kwargs['half'] = False # Ensure benchmark also knows the target precision
    precision = f"dynamic_{onnx_dynamic_quant_kwargs.get('weight_type', 'QUInt8')}"
    
    original_pt_path = Path(model_pt_path_str)
    if not original_pt_path.exists():
        print(f"Input PyTorch model not found at {original_pt_path}, downloading model...")
        # Attempt downloading the model if it doesn't exist
        model = YOLO(model_pt_path_str, task="detect")
        original_pt_path = Path(model.ckpt_path)
        print(f"Downloaded model to {original_pt_path}")

    model_stem = original_pt_path.stem # e.g., "yolov8n"

    # --- Define Directory Structure ---
    pt_model_dir = Path(models_base_dir) / "pt" / model_stem / precision
    onnx_model_dir = Path(models_base_dir) / "onnx" / model_stem / precision
    results_dir = Path(results_base_dir) / model_stem / precision

    # --- Create Directories ---
    os.makedirs(pt_model_dir, exist_ok=True)
    os.makedirs(onnx_model_dir, exist_ok=True)
    # Results dir will be created later based on dataset

    # Move the original PT model to the organized directory
    organized_pt_path = pt_model_dir / original_pt_path.name
    try:
        shutil.move(original_pt_path, organized_pt_path)
        print(f"Moved {original_pt_path} to {organized_pt_path}")
    except Exception as e:
        print(f"Error moving model {original_pt_path} to {organized_pt_path}: {e}")
        return None

    # --- Export the Copied PT model to ONNX ---
    exported_onnx_path = pt_model_dir / f"{model_stem}.onnx" # Save initial ONNX next to its PT source
    print(f"Exporting {organized_pt_path} to {exported_onnx_path}...")   
    exported_onnx_path = export_to_onnx(
        model_path=organized_pt_path,
        export_kwargs=onnx_export_kwargs,
    )
    
    # Move the exported ONNX model to the ONNX directory
    try:
        shutil.move(exported_onnx_path, onnx_model_dir / str(exported_onnx_path).split("/")[-1])
        print(f"Moved exported ONNX model to {onnx_model_dir}")
    except Exception as e:
        print(f"Error moving exported ONNX model to {onnx_model_dir}: {e}")
        return None

    # --- Preprocess the Exported ONNX model ---
    processed_onnx_path = onnx_model_dir / f"{model_stem}_processed.onnx" # Save processed model here
    try:
        # Assuming preprocess_model takes input and output paths
        preprocess_model(
            input_model_path=onnx_model_dir/f"{model_stem}.onnx",
            output_model_path=processed_onnx_path,
            preprocess_kwargs=onnx_preprocess_kwargs,
        )
        # Check if the output file was actually created
        if not processed_onnx_path.exists():
             raise FileNotFoundError("Preprocessed ONNX file was not created.")
        print("Preprocessed ONNX file was created.")
    except Exception as e:
        print(f"Error preprocessing ONNX model {exported_onnx_path}: {e}")
        return None
    
    # --- Apply Dynamic Quantization ---
    quantized_onnx_path = onnx_model_dir / f"{model_stem}_{precision}.onnx" # Save quantized model here
    dynamic_quantization(
        model_input=processed_onnx_path,
        model_output=quantized_onnx_path,
        quant_kwargs=onnx_dynamic_quant_kwargs,
    )

    # --- Perform Benchmark on the Preprocessed ONNX model ---
    dataset_yaml_path_str = onnx_benchmark_kwargs.get("data", "coco.yaml")
    dataset_stem = Path(dataset_yaml_path_str).stem # e.g., "coco" or "coco_noisy_low"

    try:
        # Load the processed ONNX model for benchmarking
        model_onnx = YOLO(str(quantized_onnx_path)) # Must use string path for YOLO constructor
        benchmark_results_obj = benchmark( # Assuming benchmark returns the results object
            model=model_onnx,
            kwargs=onnx_benchmark_kwargs, # Pass all benchmark args (includes data)
        )
        if benchmark_results_obj is None:
            raise RuntimeError("Benchmark function returned None or failed internally.")
    except Exception as e:
        print(f"Error benchmarking model {quantized_onnx_path}: {e}")
        return None

    # --- Extract and Save Results ---
    results_data = {
        "model_name": original_pt_path.stem,
        "model_type": f"{precision.upper()}_ONNX_Processed",
        "dataset": dataset_yaml_path_str,
        "hardware": get_gpu_name() if onnx_benchmark_kwargs.get("device", "cpu") == "cuda" else get_cpu_name(),
        "mAP50-95": benchmark_results_obj.results_dict.get('metrics/mAP50-95(B)', None),
        "mAP50": benchmark_results_obj.results_dict.get('metrics/mAP50(B)', None),
        "latency_ms": benchmark_results_obj.speed.get('inference', None),
        "fps": None,
    }
    if results_data["latency_ms"] is not None:
        results_data["fps"] = 1000.0 / results_data["latency_ms"]

    # Define results file path
    dataset_results_dir = results_dir / dataset_stem
    os.makedirs(dataset_results_dir, exist_ok=True)
    results_json_path = dataset_results_dir / f"{model_stem}_{precision}_benchmark.json"

    print(f"Saving benchmark results to {results_json_path}...")
    try:
        with open(results_json_path, "w") as f:
            json.dump(results_data, f, indent=4) # Use json.dump for proper formatting
        print("Benchmark results saved successfully.")
        return results_data # Return the collected data
    except Exception as e:
        print(f"Error saving results to {results_json_path}: {e}")
        return None # Indicate failure


def benchmark_yolo_static_quant(
    model_pt_path_str: str, # Changed name for clarity: Path to the original .pt file
    models_base_dir: str, # Base directory to store pt/onnx models (e.g., 'models/')
    results_base_dir: str, # Base directory to store results (e.g., 'results/')
    onnx_calibrator_kwargs: dict, # Arguments for the YOLOCalibrationDataReader
    onnx_export_kwargs: dict={}, # Arguments for the ONNX export function
    onnx_preprocess_kwargs: dict={}, # Arguments for the ONNX pre-processing function
    onnx_static_quant_kwargs: dict={}, # Arguments for static quantization
    onnx_benchmark_kwargs: dict={}, # Must contain the 'data' key for the dataset yaml
):
    """
    Benchmarks a YOLO model with Static Quantization with Calibration by exporting to ONNX, pre-processing, applying static quantization with calibration,
    and evaluating the quantized ONNX model, and saving results.

    Args:
        model_pt_path_str (str): Path to the original YOLOv8 PyTorch (.pt) model file.
        models_base_dir (str): Base directory to store organized model files (pt and onnx).
        results_base_dir (str): Base directory to store organized benchmark results.
        onnx_calibrator_kwargs (dict): Arguments for the YOLOCalibrationDataReader.
        onnx_export_kwargs (dict): Arguments for the ONNX export function.
        onnx_preprocess_kwargs (dict): Arguments for the ONNX pre-processing function.
        onnx_static_quant_kwargs (dict): Arguments for static quantization.
        onnx_benchmark_kwargs (dict): Arguments for the benchmark function (MUST include 'data' key).
    
    Returns:
        dict: A dictionary containing benchmark results including mAP50, mAP50-95, latency, and FPS.
    """
    # --- Ensure Consistency for 'half', should be False for static quantization ---
    onnx_export_kwargs['half'] = False
    onnx_benchmark_kwargs['half'] = False
    precision = f"static_{onnx_static_quant_kwargs.get('weight_type', 'QInt8')}_{onnx_static_quant_kwargs.get('activation_type', 'QInt8')}"
    
    original_pt_path = Path(model_pt_path_str)
    if not original_pt_path.exists():
        print(f"Input PyTorch model not found at {original_pt_path}, downloading model...")
        # Attempt to download the model if it doesn't exist
        model = YOLO(model_pt_path_str, task="detect")
        original_pt_path = Path(model.ckpt_path)
        print(f"Downloaded model to {original_pt_path}")

    model_stem = original_pt_path.stem # e.g., "yolov8n"

    # --- Define Directory Structure ---
    pt_model_dir = Path(models_base_dir) / "pt" / model_stem / precision
    onnx_model_dir = Path(models_base_dir) / "onnx" / model_stem / precision
    results_dir = Path(results_base_dir) / model_stem / precision

    # --- Create Directories ---
    os.makedirs(pt_model_dir, exist_ok=True)
    os.makedirs(onnx_model_dir, exist_ok=True)
    # Results dir will be created later based on dataset

    # Move the original PT model to the organized directory
    organized_pt_path = pt_model_dir / original_pt_path.name
    try:
        shutil.move(original_pt_path, organized_pt_path)
        print(f"Moved {original_pt_path} to {organized_pt_path}")
    except Exception as e:
        print(f"Error moving model {original_pt_path} to {organized_pt_path}: {e}")
        return None

    # --- Export the Copied PT model to ONNX ---
    exported_onnx_path = pt_model_dir / f"{model_stem}.onnx" # Save initial ONNX next to its PT source
    print(f"Exporting {organized_pt_path} to {exported_onnx_path}...")
    exported_onnx_path = export_to_onnx(
        model_path=organized_pt_path,
        export_kwargs=onnx_export_kwargs,
    )
    
    # Move the exported ONNX model to the ONNX directory
    try:
        shutil.move(exported_onnx_path, onnx_model_dir / str(exported_onnx_path).split("/")[-1])
        print(f"Moved exported ONNX model to {onnx_model_dir}")
    except Exception as e:
        print(f"Error moving exported ONNX model to {onnx_model_dir}: {e}")
        return None

    # --- Preprocess the Exported ONNX model ---
    processed_onnx_path = onnx_model_dir / f"{model_stem}_processed.onnx" # Save processed model here
    try:
        # Assuming preprocess_model takes input and output paths
        preprocess_model(
            input_model_path=onnx_model_dir/f"{model_stem}.onnx",
            output_model_path=processed_onnx_path,
            preprocess_kwargs=onnx_preprocess_kwargs,
        )
        # Check if the output file was actually created
        if not processed_onnx_path.exists():
             raise FileNotFoundError("Preprocessed ONNX file was not created.")
        print("Preprocessed ONNX file was created.")
    except Exception as e:
        print(f"Error preprocessing ONNX model {exported_onnx_path}: {e}")
        return None
    
    # --- Apply Static Quantization ---
    onnx_calibrator_kwargs['model_path'] = str(processed_onnx_path) # Ensure the model path is set for calibration
    try:
        print(f"Initializing YOLOCalibrationDataReader...")
        calibration_datareader = YOLOCalibrationDataReader(**onnx_calibrator_kwargs)
        # calibration_datareader = ImageCalibrationDataReader(image_paths=onnx_calibrator_kwargs['image_folder_or_list'])
    except Exception as e:
        print(f"Error initializing YOLOCalibrationDataReader: {e}")
        return None
    
    quantized_onnx_path = onnx_model_dir / f"{model_stem}_{precision}.onnx" # Save quantized model here
    static_quantization(
        model_input=processed_onnx_path,
        model_output=quantized_onnx_path,
        calibration_data=calibration_datareader,
        quant_kwargs=onnx_static_quant_kwargs,
    )

    # --- Perform Benchmark on the Preprocessed | Quantized ONNX model ---
    dataset_yaml_path_str = onnx_benchmark_kwargs.get("data", "coco.yaml")
    dataset_stem = Path(dataset_yaml_path_str).stem # e.g., "coco" or "coco_noisy_low"

    try:
        # Load the processed ONNX model for benchmarking
        model_onnx = YOLO(str(quantized_onnx_path)) # Must use string path for YOLO constructor
        benchmark_results_obj = benchmark( # Assuming benchmark returns the results object
            model=model_onnx,
            kwargs=onnx_benchmark_kwargs, # Pass all benchmark args (includes data)
        )
        if benchmark_results_obj is None:
            raise RuntimeError("Benchmark function returned None or failed internally.")
    except Exception as e:
        print(f"Error benchmarking model {quantized_onnx_path}: {e}")
        return None

    # --- Extract and Save Results ---
    results_data = {
        "model_name": original_pt_path.stem,
        "model_type": f"{precision.upper()}_ONNX_Processed",
        "dataset": dataset_yaml_path_str,
        "hardware": get_gpu_name() if onnx_benchmark_kwargs.get("device", "cpu") == "cuda" else get_cpu_name(),
        "mAP50-95": benchmark_results_obj.results_dict.get('metrics/mAP50-95(B)', None),
        "mAP50": benchmark_results_obj.results_dict.get('metrics/mAP50(B)', None),
        "latency_ms": benchmark_results_obj.speed.get('inference', None),
        "fps": None,
    }
    if results_data["latency_ms"] is not None:
        results_data["fps"] = 1000.0 / results_data["latency_ms"]

    # Define results file path
    dataset_results_dir = results_dir / dataset_stem
    os.makedirs(dataset_results_dir, exist_ok=True)
    results_json_path = dataset_results_dir / f"{model_stem}_{precision}_benchmark.json"

    print(f"Saving benchmark results to {results_json_path}...")
    try:
        with open(results_json_path, "w") as f:
            json.dump(results_data, f, indent=4) # Use json.dump for proper formatting
        print("Benchmark results saved successfully.")
        return results_data # Return the collected data
    except Exception as e:
        print(f"Error saving results to {results_json_path}: {e}")
        return None # Indicate failure