from pathlib import Path
from onnxruntime.quantization import quantize_static, CalibrationDataReader

def static_quant(model_input:str|Path, 
                 model_output:str|Path,
                 calibration_data: CalibrationDataReader):
    quantize_static(
        
    )