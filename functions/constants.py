from onnxruntime.quantization import QuantType
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
    "save_as_external_data": False,  # Not needed for models smaller than 2GB
}

# ONNX Dynamic Quantization kwargs #
default_onnx_dynamic_quant_kwargs = {
    'op_types_to_quantize': None, # Specifies which operator types should have their weights quantized offline. Dynamic quantization primarily targets operators like Conv, MatMul, LSTM, GRU where weights are static.
    'per_channel': True, # Quantizes weights per-channel (True) or per-tensor (False)
    'reduce_range': False, # Uses 7-bit range for weights instead of 8-bit.
    'weight_type': QuantType.QUInt8, # Target data type for the weights that are quantized offline. (QUInt8 is the only supported)
    'nodes_to_quantize': None, # Explicit list of node names whose weights should be quantized.
    'nodes_to_exclude': None, # Explicit list of node names whose weights should not be quantized. (kept in FP32)
    'use_external_data_format': False, # If True, the model will be saved in external data format. This is useful for large models (>2GB) to avoid loading the entire model into memory at once.
    'extra_options': None, # Additional options for the quantization process. This is a dictionary that can contain various settings depending on the specific requirements of the quantization process.
}

# Benchmarking constants #
default_benchmark_kwargs = {
    "imgsz": 640,
    "device": "cpu",
    "iou": 0.6,
    "conf": 0.001,
    "max_det": 300,
    "split": "val",
    "batch": 1,
    "half": False,
    "rect": True,
    "dnn": False,
    "workers": 8,
    "augment": False,
    "agnostic_nms": False,
    "classes": None,
    "single_cls": False,
    "data": "coco.yaml",
    "plots": False,
    "save_json": False,
    "verbose": False,
}