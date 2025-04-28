import torch
from onnxruntime.quantization import QuantType, QuantFormat, CalibrationMethod
from functions.benchmark_models import benchmark_yolo_fp
from functions.benchmark_models import benchmark_yolo_static_quant
from functions.benchmark_models import benchmark_yolo_dynamic_quant
from helpers.read_json_for_calibration import read_json_for_calibration

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    model_list = [
        'yolov8n',
        'yolo11n',
        'yolo12n',
    ]
    
    print("Starting benchmark...")
    
    for model in model_list:
        
        print(f"Benchmarking {model}...")
        
        print("Full Precision FP32")
        benchmark_yolo_fp(
            model_pt_path_str=model,
            models_base_dir='/home/omni/Programming/QRID/QRID/models',
            results_base_dir='/home/omni/Programming/QRID/QRID/results',
            onnx_export_kwargs={
                "device": device,
            },
            onnx_benchmark_kwargs={
                "device": device,
            }
        )

        print("Half Precision FP16")
        benchmark_yolo_fp(
            model_pt_path_str=model,
            models_base_dir='/home/omni/Programming/QRID/QRID/models',
            results_base_dir='/home/omni/Programming/QRID/QRID/results',
            half=True,
            onnx_export_kwargs={
                "device": device,
            },
            onnx_benchmark_kwargs={
                "device": device,
            }
        )
        
        print("Dynamic Quantization")
        benchmark_yolo_dynamic_quant(
            model_pt_path_str=model,
            models_base_dir='/home/omni/Programming/QRID/QRID/models',
            results_base_dir='/home/omni/Programming/QRID/QRID/results',
            onnx_export_kwargs={
                "device": device,
            },
            onnx_benchmark_kwargs={
                "device": device,
            },
            onnx_dynamic_quant_kwargs={
                "weight_type": QuantType.QUInt8,
            }
        )
        
        print("Static Quantization")
        benchmark_yolo_static_quant(
            model_pt_path_str=model,
            models_base_dir='/home/omni/Programming/QRID/QRID/models',
            results_base_dir='/home/omni/Programming/QRID/QRID/results',
            calibration_image_paths=read_json_for_calibration("/home/omni/Programming/QRID/QRID/imgs_calibrated_for_ptq/calibration_files_sampled_final.json"),
            onnx_export_kwargs={
                "device": "cuda" if torch.cuda.is_available() else "cpu",
            },
            onnx_benchmark_kwargs={
                "device": "cuda" if torch.cuda.is_available() else "cpu",
            },
            onnx_static_quant_kwargs = dict(
                activation_type=QuantType.QInt8, 
                weight_type=QuantType.QInt8,       
                quant_format=QuantFormat.QDQ,      
                per_channel=True,
                calibrate_method=CalibrationMethod.MinMax,
            )
        )
        
        print("Benchmarking completed for", model)
        
        print("=====================================")
    
    print("All benchmarks completed.")

if __name__ == "__main__":
    main()