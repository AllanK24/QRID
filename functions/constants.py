# Export constants to ONNX for YOLO
default_onnx_export_kwargs = {
    "imgsz": 640,
    "half": False,
    "dynamic": False,
    "simplify": True,
    "opset": None,
    "nms": False,
    "batch": 1,
    "device": "cpu",
}

# ONNX Pre-Process kwargs #
default_onnx_preprocess_kwargs = {
    "skip_optimization": False,      # DO perform model optimization (Fuse Conv+BN, etc.)
    "skip_onnx_shape": False,        # DO perform standard ONNX shape inference
    "skip_symbolic_shape": True,     # SKIP symbolic shape inference (less relevant for CNNs)
    "auto_merge": False,             # Default, irrelevant since symbolic is skipped
    "int_max": 2**31 - 1,          # Default, irrelevant since symbolic is skipped
    "guess_output_rank": False,      # Default, rely on explicit inference
    "verbose": 1,                    # Show warnings, good for monitoring
    "save_as_external_data": False,  # Not needed for YOLOv8n size
}
