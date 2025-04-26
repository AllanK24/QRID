weight type, activation type supports:
* QInt8
* QInt8, QUInt8
* QUInt8, QUInt8
Miserable difference, but the best one is QInt8 for both weight and activation

**To Check out**:
calibrate_method
what calibrator data reader is better: YOLOCalibrationDataReader vs ImageCalibrationDataReader

test the automatic node exclude function on newer yolo models: yolo11 and yolo12