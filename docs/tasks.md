weight type, activation type supports:
* QInt8
* QInt8, QUInt8
* QUInt8, QUInt8
Miserable difference, but the best one is QInt8 for both weight and activation

Ultralytics benchmark function uses:
* model.val method with batch_size=1
* Use TensorRT for Static Quantization -> it's 3x faster than dynamic quantization via ONNX