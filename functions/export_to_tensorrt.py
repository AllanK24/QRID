from ultralytics import YOLO
from functions.constants import default_tensorrt_export_kwargs

def export_to_tensorrt(model: YOLO, kwargs:dict):
    try:
        kwatgs = {**default_tensorrt_export_kwargs, **kwargs}
        exported_path = model.export(
            model=model,
            **kwatgs,
        )
    except Exception as e:
        print(f"Error exporting to TensorRT: {e}")
        return None
    return exported_path