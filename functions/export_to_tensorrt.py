from ultralytics import YOLO
from functions.constants import default_tensorrt_export_kwargs

def export_to_tensorrt(model: YOLO, kwargs:dict=default_tensorrt_export_kwargs) -> str:
    try:
        kwargs = {**default_tensorrt_export_kwargs, **kwargs}
        exported_path = model.export(
            **kwargs,
        )
    except Exception as e:
        print(f"Error exporting to TensorRT: {e}")
        return None
    return exported_path