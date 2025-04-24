from ultralytics import YOLO
from pathlib import Path
from functions.constants import default_onnx_export_kwargs
def export_to_onnx(model_path:str | Path, export_args:dict=default_onnx_export_kwargs, task="detect"):
    """This function exports a YOLO model to ONNX format. It takes the model path and export arguments as input, and uses the YOLO library to perform the export. The function also handles exceptions that may occur during the loading and exporting process.

    Args:
        model_path (str | Path): The path to the YOLO model file. This should be a valid YOLO model file.
        export_args (dict): A dictionary of arguments to be passed to the YOLO export function. This allows for customization of the ONNX export process.
        task (str, optional): The task for which the model is being exported. This can be "detect", "segment", or "pose". The default value is "detect". This argument is passed to the YOLO constructor to specify the type of model being used. It determines how the model will be loaded and exported.
    Returns:
        None: This function does not return any value. It performs the export process and prints messages indicating the success or failure of the operation.
    Raises:
        Exception: If there is an error during the loading or exporting process, an exception will be raised. This could be due to issues with the model path, export arguments, or other factors.
    """
    try:
        # Load the YOLOv8 model
        print(f"Loading model {model_path}...")
        model = YOLO(model_path, task=task)
        print(f"Model {model_path} loaded successfully.")
    except Exception as e:
        print(f"Error loading model {model_path}: {e}. Please check the model name and try again.")
        return
    try:
        # Merge default export arguments with user-provided arguments
        kwargs = {**default_onnx_export_kwargs, **export_args}
        # Export the model to ONNX format
        model.export(format="onnx", **kwargs)
        print(f"Model {model_path} exported to ONNX format successfully.")
    except Exception as e:
        print(f"Error exporting model {model_path} to ONNX: {e}. Please check the export arguments and try again.")
        return

# Example usage
if __name__=="__main__":
    export_to_onnx(
        model_path="yolov8n.pt",
        export_args={
            "device": "cuda",
        },
    )