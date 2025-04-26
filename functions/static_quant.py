import cv2 # Using OpenCV for image loading/preprocessing
import numpy as np
from pathlib import Path
import onnxruntime as ort
from functions.constants import default_onnx_static_quant_kwargs
from onnxruntime.quantization import quantize_static, CalibrationDataReader

class YOLOCalibrationDataReader(CalibrationDataReader):
    def __init__(self, image_paths):
        self.image_paths = image_paths
        self.idx = 0
        self.input_name = "images"

    def preprocess(self, frame):
        # Same preprocessing that you do before feeding it to the model
        frame = cv2.imread(frame)
        X = cv2.resize(frame, (640, 640))
        image_data = np.array(X).astype(np.float32) / 255.0  # Normalize to [0, 1] range
        image_data = np.transpose(image_data, (2, 0, 1))  # (H, W, C) -> (C, H, W)
        image_data = np.expand_dims(image_data, axis=0)  # Add batch dimension
        return image_data

    def get_next(self):
        # method to iterate through the data set
        if self.idx >= len(self.image_paths):
            return None

        image_path = self.image_paths[self.idx]
        input_data = self.preprocess(image_path)
        self.idx += 1
        return {self.input_name: input_data}

def static_quantization(model_input:str|Path, 
                 model_output:str|Path,
                 calibration_data: CalibrationDataReader,
                 quant_kwargs: dict = default_onnx_static_quant_kwargs):
    
    # Merge default quantization arguments with user-provided arguments
    quant_kwargs = {**default_onnx_static_quant_kwargs, **quant_kwargs}
    print(f"Performing static quantization on model {model_input}...")
    try:
        quantize_static(
            model_input=model_input,
            model_output=model_output,
            calibration_data_reader=calibration_data,
            **quant_kwargs,
        )
        print(f"Model {model_input} quantized successfully.")
    except Exception as e:
        print(f"Error during static quantization of model {model_input}: {e}. Please check the model path and quantization arguments.")
        return