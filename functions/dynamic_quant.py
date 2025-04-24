from pathlib import Path
from functions.constants import default_onnx_dynmic_quant_kwargs
from onnxruntime.quantization import quantize_dynamic

def dynamic_quantization(model:str|Path,
                         model_output:str|Path,
                         kwargs:dict=default_onnx_dynmic_quant_kwargs) -> None:
    """Perform dynamic quantization on the given model. This function is used to reduce the size of the model and improve inference speed by quantizing the weights of the model.

    Args:
        model (str | Path): Model to be quantized. This should be a valid ONNX model file.
        model_output (str | Path): Output path for the quantized model. This is where the quantized model will be saved.
        kwargs (dict, optional): A dictionary of arguments to be passed to the quantize_dynamic function. This allows for customization of the quantization process. Defaults to default_onnx_dynmic_quant_kwargs.
    
    Returns:
        None: This function does not return any value. It saves the quantized model to the specified output path.
        
    Raises:
        Exception: If there is an error during the quantization process, an exception will be raised. This could be due to issues with the model path, output path, or the quantization arguments.
    """
    try:
        # Merge default quantization arguments with user-provided arguments
        kwargs = {**default_onnx_dynmic_quant_kwargs, **kwargs}
        print(f"Performing dynamic quantization on model {model}...")
        quantize_dynamic(model, model_output, **kwargs)
        print(f"Model {model} quantized successfully.")
    except Exception as e:
        print(f"Error during dynamic quantization of model {model}: {e}. Please check the model path and quantization arguments.")
        return