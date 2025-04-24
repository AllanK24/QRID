import torch
from ultralytics.models import YOLO
def benchmark(model, val_data):
    if isinstance(model, YOLO):
        return
    validation_results = model.val(data="coco.yaml", imgsz=640, batch=16, device="0")