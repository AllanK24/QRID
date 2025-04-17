from pathlib import Path
from constants import default_onnx_preprocess_kwargs
from onnxruntime.quantization.preprocess import quant_pre_process

def preprocess_model(input_model_path: str | Path, output_model_path: str | Path, preprocess_args:dict=default_onnx_preprocess_kwargs) -> None:
    """Preprocess the model using ONNX quantization preprocess. This function is used to optimize the model for inference by applying various preprocessing techniques.
    This includes optimizations such as fusing Conv+BN layers, performing shape inference, and other optimizations that can improve the performance of the model during inference.

    Args:
        input_model_path (str | Path): The path to the input model file. This should be a valid ONNX model file.
        output_model_path (str | Path): The path where the preprocessed model will be saved.
        preprocess_args (dict, optional): A dictionary of arguments to be passed to the quant_pre_process function. This allows for customization of the preprocessing steps. Defaults to default_onnx_preprocess_kwargs.
    Returns:
        None: This function does not return any value. It saves the preprocessed model to the specified output path.
    Raises:
        Exception: If there is an error during the preprocessing step, an exception will be raised. This could be due to issues with the input model path, output model path, or the preprocessing arguments.
    """
    try:
        kwargs = {**default_onnx_preprocess_kwargs, **preprocess_args}
        print(f"Preprocessing model {input_model_path}...")
        quant_pre_process(input_model_path, output_model_path, **kwargs)
        print(f"Model {input_model_path} preprocessed successfully.")
    except Exception as e:
        print(f"Error preprocessing model {input_model_path}: {e}. Please check the input model path and preprocess arguments.")
        return
    
    
# Example usage
if __name__=="__main__":
    preprocess_model(
        input_model_path="yolov8n.onnx",
        output_model_path="yolov8n_preprocessed.onnx",
    )