from ultralytics import YOLO
def export_to_onnx(model_name, export_args, task="detect"):
    try:
        # Load the YOLOv8 model
        print(f"Loading model {model_name}...")
        model = YOLO(model_name, task=task)
        print(f"Model {model_name} loaded successfully.")
    except Exception as e:
        print(f"Error loading model {model_name}: {e}. Please check the model name and try again.")
        return
    try:
        # Export the model to ONNX format
        model.export(format="onnx", **export_args)
        print(f"Model {model_name} exported to ONNX format successfully.")
    except Exception as e:
        print(f"Error exporting model {model_name} to ONNX: {e}. Please check the export arguments and try again.")
        return