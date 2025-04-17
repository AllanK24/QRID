import torch
from ultralytics.models import YOLO
def benchmark(model, val_data):
    if isinstance(model, YOLO):
        return
    model.val()