from pathlib import Path
from functions.constants import default_onnx_static_quant_kwargs
from onnxruntime.quantization import quantize_static, CalibrationDataReader

def static_quant(model_input:str|Path, 
                 model_output:str|Path,
                 calibration_data: CalibrationDataReader,
                 static_quant_kwargs: dict = default_onnx_static_quant_kwargs):
    
    # Merge default quantization arguments with user-provided arguments
    static_quant_kwargs = {**default_onnx_static_quant_kwargs, **static_quant_kwargs}
    print(f"Performing static quantization on model {model_input}...")
    try:
        quantize_static(
            model_input=model_input,
            model_output=model_output,
            calibration_data_reader=calibration_data
            **static_quant_kwargs,
        )
        print(f"Model {model_input} quantized successfully.")
    except Exception as e:
        print(f"Error during static quantization of model {model_input}: {e}. Please check the model path and quantization arguments.")
        return