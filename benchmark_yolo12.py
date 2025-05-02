from functions.benchmark_models import benchmark_yolo_fp, benchmark_yolo_dynamic_quant, benchmark_yolo_static_quant_tensorrt

model_list = [
    "yolo12n",
    "yolo12s",
    "yolo12m",
    "yolo12l",
    "yolo12x",
]

calibration_sets = [
    "/home/omni/Programming/QRID/QRID/calibration_sets/coco_calib_clean/data.yaml",
    "/home/omni/Programming/QRID/QRID/calibration_sets/coco_calib_mixed/data.yaml",
]

validation_sets = [
    "coco.yaml",
    '/home/omni/Programming/QRID/QRID/validation_datasets/coco_val_blurry_low/data_blurry_low.yaml',
    "/home/omni/Programming/QRID/QRID/validation_datasets/coco_val_blurry_medium/data_blurry_medium.yaml",
    "/home/omni/Programming/QRID/QRID/validation_datasets/coco_val_contrast_low/data_contrast_low.yaml",
    "/home/omni/Programming/QRID/QRID/validation_datasets/coco_val_jpeg_heavy/data_jpeg_heavy.yaml",
    "/home/omni/Programming/QRID/QRID/validation_datasets/coco_val_noisy_low/data_noisy_low.yaml",
    "/home/omni/Programming/QRID/QRID/validation_datasets/coco_val_noisy_medium/data_noisy_medium.yaml",
    "/home/omni/Programming/QRID/QRID/validation_datasets/coco_val_mixed_degrad_50pct/data_coco_val_mixed_degrad_50pct.yaml",
]

models_base_dir = "/home/omni/Programming/QRID/QRID/models"
results_base_dir = "/home/omni/Programming/QRID/QRID/results"

def main():
    
    print("Starting YOLOv12 Benchmarking...")
    
    for model in model_list:
        print(f"Benchmarking model: {model}")
        for validation_set in validation_sets:
            print(f"Benchmarking model: {model} on validation set: {validation_set}")
            print("Benchmarking FP32...")
            benchmark_yolo_fp(
                model_pt_path_str=model,
                models_base_dir=models_base_dir,
                results_base_dir=results_base_dir,
                half=False,
                benchmark_kwargs={
                    "data": validation_set,
                    "device": "cuda",
                }
            )
            print("Benchmarking FP32 finished successfully.")
            print("Benchmarking FP16...")
            benchmark_yolo_fp(
                model_pt_path_str=model,
                models_base_dir=models_base_dir,
                results_base_dir=results_base_dir,
                half=True,
                benchmark_kwargs={
                    "data": validation_set,
                    "device": "cuda",
                }
            )
            print("Benchmarking FP16 finished successfully.")
            print("Benchmarking dynamic quantization...")
            benchmark_yolo_dynamic_quant(
                model_pt_path_str=model,
                models_base_dir=models_base_dir,
                results_base_dir=results_base_dir,
                onnx_export_kwargs={
                    "device": "cuda",
                },
                benchmark_kwargs={
                    "data": validation_set,
                    "device": "cuda"
                }
            )
            print("Benchmarking dynamic quantization finished successfully.")
            
            print(f"Benchmarked {model} on {validation_set} finished successfully.")
        
        print(f"Finished benchmarking {model} on all validation sets.")
    
    print("Finished benchmarking FP32, FP16 and Dynamic Quantization Yolo on all validation sets.")    
        
    print(f"Starting benchmarking for static quantization with TensorRT...")
    for model in model_list:
        print(f"Benchmarking model: {model}")
        for calibration_set in calibration_sets:
            print(f"Benchmarking model: {model} on calibration set: {calibration_set}")
            for validation_set in validation_sets:
                print(f"Benchmarking on validation set: {validation_set}")
                benchmark_yolo_static_quant_tensorrt(
                    model_pt_path_str=model,
                    models_base_dir=models_base_dir,
                    results_base_dir=results_base_dir,
                    tensorrt_export_kwargs={
                        "data": calibration_set,
                    },
                    benchmark_kwargs={
                        "data": validation_set,
                        "device": "cuda",
                    }
                )
                print(f"Benchmarking {model} on {validation_set} finished successfully.")
            print(f"Finished benchmarking model on calibration set: {calibration_set} and all validation sets.")
        print(f"Finished benchmarking {model} on all calibration sets.")
    
    print("Finished benchmarking static quantization with TensorRT on all models, calibration sets and validation sets.")
    
    print("All benchmarking completed successfully.")
    
def test_benchmark_functions():
    model = "yolo12n"
    validation_set = validation_sets[0]
    calibration_set = calibration_sets[0]
    print(f"Benchmarking model: {model} on validation set: {validation_set}")
    print("Benchmarking FP32...")
    benchmark_yolo_fp(
        model_pt_path_str=model,
        models_base_dir=models_base_dir,
        results_base_dir=results_base_dir,
        half=False,
        benchmark_kwargs={
            "data": validation_set,
            "device": "cuda",
        }
    )
    print("Benchmarking FP32 finished successfully.")
    print("Benchmarking FP16...")
    benchmark_yolo_fp(
        model_pt_path_str=model,
        models_base_dir=models_base_dir,
        results_base_dir=results_base_dir,
        half=True,
        benchmark_kwargs={
            "data": validation_set,
            "device": "cuda",
        }
    )
    print("Benchmarking FP16 finished successfully.")
    print("Benchmarking dynamic quantization...")
    benchmark_yolo_dynamic_quant(
        model_pt_path_str=model,
        models_base_dir=models_base_dir,
        results_base_dir=results_base_dir,
        onnx_export_kwargs={
            "device": "cuda",
        },
        benchmark_kwargs={
            "data": validation_set,
            "device": "cuda"
        }
    )
    print("Benchmarking dynamic quantization finished successfully.")
    
    print(f"Starting static quantization with TensorRT...")
    print(f"Benchmarking on validation set: {validation_set}")
    benchmark_yolo_static_quant_tensorrt(
        model_pt_path_str=model,
        models_base_dir=models_base_dir,
        results_base_dir=results_base_dir,
        tensorrt_export_kwargs={
            "data": calibration_set,
        },
        benchmark_kwargs={
            "data": validation_set,
            "device": "cuda",
        }
    )
    print(f"Finished static quantization with TensorRT on {model} on {validation_set} successfully with calibration set {calibration_set}.")
    
if __name__ == "__main__":
    main()