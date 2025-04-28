weight type, activation type supports:
* QInt8
* QInt8, QUInt8
* QUInt8, QUInt8
Miserable difference, but the best one is QInt8 for both weight and activation

Ultralytics benchmark function uses:
* model.val method with batch_size=1
* Use TensorRT for Static Quantization -> it's 3x faster than dynamic quantization via ONNX

I'm not sure if we can use TensorRT for dynamic quantization for Yolo models, but ultralytics api only supports static quantization.

**Possible Solution**:
1. Apply Dynamic Quantization in ONNX -> then convert the dynamically quantized ONNX model into TensorRT Engine -> load the model using YOLO class -> use model.val() for validation
2. For static quantization: export the Yolo model directly to TensorRT using calibration and int8 -> static quantization will be automatically applied -> load the `engine` model using Yolo class -> use model.val() for validation