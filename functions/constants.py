from onnxruntime.quantization import QuantType, QuantFormat, CalibrationMethod
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

# ONNX Static Quantization kwargs #

#- Extra options for ONNX Static Quantization -#
common_extra_options_static_quant = {
    "ActivationSymmetric": True, # The documentation explicitly states symmetric activation quantization is required if targeting GPU/TRT execution providers. Even for CPU, it's often compatible and simplifies the quantization scheme (zero-point is often fixed to 0). While asymmetric might offer slightly better pure CPU accuracy in some cases, using symmetric provides broader compatibility and is standard practice for INT8 deployment.
    "WeightSymmetric": True, # This is the default (True) and standard practice. Symmetric weight quantization is generally assumed and often required by hardware backends.
    "EnableSubgraph": False, # Not typically needed unless your model has explicit subgraphs that also need quantization (less common in standard CNNs like YOLOv8).
    'ForceQuantizeNoInputCheck': False, # Avoids potentially quantizing operators like MaxPool unnecessarily if their inputs aren't already quantized, which could sometimes lead to unexpected behavior or accuracy drops.
    'MatMulConstBOnly': False, # Allows quantization of MatMul layers even if the weight tensor isn't technically a constant initializer (though it usually is in inference models).
    'AddQDQPairToWeight': False, # Standard approach is to quantize the weight tensor itself and only insert the DequantizeLinear node. This results in smaller model files. Keeping FP32 weights and adding Q+DQ nodes is less common.
}

default_onnx_static_quant_kwargs = {
    "quant_format": QuantFormat.QDQ, # Explicitly recommended, modern standard, better compatibility with various execution providers (especially accelerators), easier debugging.
    "op_types_to_quantize": None, # Specifies which operator types should be quantized. If None, all supported operators will be quantized.
    "per_channel": True, # Quantizes weights per-channel (True) or per-tensor (False)
    "reduce_range": False, # Uses 7-bit range for weights instead of 8-bit.
    "activation_type": QuantType.QInt8, # Target standard signed INT8 activations. Needed for S8S8 scheme.
    "weight_type": QuantType.QInt8, # Target standard signed INT8 weights. Matches activation_type for S8S8 scheme, required for GPU/TRT compatibility and generally recommended.
    "nodes_to_quantize": None, # Not needed for quantizing the whole model by type.
    "nodes_to_exclude": None, # Explicit list of node names whose weights should not be quantized. (kept in FP32)
    "use_external_data_format": False, # If True, the model will be saved in external data format. This is useful for large models (>2GB) to avoid loading the entire model into memory at once.
    "calibration_method": CalibrationMethod.MinMax, # Simpler and faster calibration method. Start with this. If you see accuracy issues potentially caused by outliers, you can experiment with CalibrationMethod.Entropy later.
    "calibration_providers": ['CPUExecutionProvider'], # List of execution providers to use for calibration. This is important for compatibility with different hardware accelerators.
    "extra_options": common_extra_options_static_quant.copy()
}
