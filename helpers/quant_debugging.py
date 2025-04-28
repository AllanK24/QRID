import json
import time
from pathlib import Path
from typing import Dict, Tuple, List

import numpy as np
import onnxruntime as ort
from onnxruntime.quantization.qdq_loss_debug import (
    create_weight_matching,
    modify_model_output_intermediate_tensors,
    collect_activations,
    create_activation_matching,
)


##########################
# Utility helpers
##########################

def _mse(a: np.ndarray, b: np.ndarray) -> float:
    """Mean‑squared‑error between two tensors cast to fp32."""
    return float(np.mean((a.astype(np.float32) - b.astype(np.float32)) ** 2))


def build_random_feed(session: ort.InferenceSession, batch: int = 1, img: int = 640) -> Dict[str, np.ndarray]:
    """Generate a dummy input tensor compatible with the first model input."""
    inp = session.get_inputs()[0]
    tensor = np.random.rand(batch, 3, img, img).astype(np.float32)
    return {inp.name: tensor}


##########################
# Profiling helpers
##########################

def profile_model(model: str, providers: List[str], feed: Dict[str, np.ndarray], *, warmup=5, repeat=50) -> Tuple[float, str]:
    """Return (average_latency_sec, profile_file_path)."""
    so = ort.SessionOptions()
    so.enable_profiling = True
    session = ort.InferenceSession(model, so, providers=providers)

    # --- warm‑up ---
    for _ in range(warmup):
        session.run(None, feed)

    # --- timed run ---
    start = time.perf_counter()
    for _ in range(repeat):
        session.run(None, feed)
    latency = (time.perf_counter() - start) / repeat

    profile_file = session.end_profiling()
    return latency, profile_file


def _get_events(profile_json):
    """
    Return the list of node events regardless of ORT profiling format.
    """
    if isinstance(profile_json, list):                     # ORT ≥ 1.17
        return profile_json
    if "events" in profile_json:                          # ORT ≤ 1.16
        return profile_json["events"]
    if "traceEvents" in profile_json:                     # very old traces
        return profile_json["traceEvents"]
    raise ValueError("Unrecognized ORT profile schema")

def top_slowest_ops(profile_file: str, k: int = 10):
    """
    Print top-k slowest operator nodes from an ORT JSON profile.
    Compatible with all ORT versions.
    """
    with open(profile_file) as f:
        data = json.load(f)

    events = _get_events(data)
    node_events = [e for e in events if e.get("cat") == "Node"]
    node_events.sort(key=lambda x: x["dur"], reverse=True)

    print(f"Top-{k} slowest ops in {Path(profile_file).name} (dur = µs):")
    for e in node_events[:k]:
        print(f"  {e['name']:<40} {e['dur']/1000:.3f} ms")
        

##########################
# Activation comparison helpers
##########################

def collect_all_activations(model: str, reader):
    """Augment model to expose intermediates and collect activations."""
    aug_model = str(Path(model).with_suffix(".activ.onnx"))
    modify_model_output_intermediate_tensors(model, aug_model)
    return collect_activations(aug_model, reader)


def compare_activations(fp32_acts: Dict[str, np.ndarray], q_acts: Dict[str, np.ndarray], topk: int = 10):
    mapping = create_activation_matching(fp32_acts, q_acts)
    diffs = []
    for k_fp32, k_q in mapping.items():
        diff = _mse(fp32_acts[k_fp32], q_acts[k_q])
        diffs.append((k_fp32, diff))
    diffs.sort(key=lambda x: x[1], reverse=True)
    print(f"\nTop‑{topk} activation MSEs (layer : mse):")
    for name, err in diffs[:topk]:
        print(f"  {name:<50} {err:.6f}")


##########################
# Simple data reader wrapper
##########################
from onnxruntime.quantization import CalibrationDataReader

class SingleDataReader(CalibrationDataReader):
    """
    Minimal reader that supplies ONE input batch.

    ORT ≥1.16 expects:
      • __iter__ returning self
      • __next__ raising StopIteration
      • (optional) get_next() for older APIs
    """
    def __init__(self, feed_dict):
        self.feed = feed_dict
        self._yielded = False

    # Iterator protocol ---------------------------------
    def __iter__(self):
        self.rewind()          # start fresh each time
        return self

    def __next__(self):
        if self._yielded:
            raise StopIteration
        self._yielded = True
        return self.feed

    # Older ORT still calls this -------------------------
    def get_next(self):
        if self._yielded:
            return None
        self._yielded = True
        return self.feed

    def rewind(self):
        self._yielded = False
        

##########################
# Main routine
##########################

def main(fp32_model: str, quant_model: str, providers: List[str]):
    # build a dummy input just to test; replace with real pre‑processed image batches for accuracy‑sensitive work
    base_session = ort.InferenceSession(fp32_model, providers=providers)
    feed = build_random_feed(base_session)

    # --- latency & operator profile ---
    fp32_lat, fp32_prof = profile_model(fp32_model, providers, feed)
    quant_lat, quant_prof = profile_model(quant_model, providers, feed)

    print("\n=== Latency (ms) ===")
    print(f"FP32 : {fp32_lat*1000:.3f} ms  |  Dynamic‑Q : {quant_lat*1000:.3f} ms  |  Δ : {(quant_lat-fp32_lat)*1000:.2f} ms")

    top_slowest_ops(fp32_prof)
    top_slowest_ops(quant_prof)

    # --- activation mismatch ---
    reader = SingleDataReader(feed)
    fp32_acts = collect_all_activations(fp32_model, reader)
    reader.rewind()
    quant_acts = collect_all_activations(quant_model, reader)

    compare_activations(fp32_acts, quant_acts)

    # --- optional: weight diff (rarely tied to perf but easy to do) ---
    weights = create_weight_matching(fp32_model, quant_model)
    print(f"\nMatched {len(weights)} weight tensors between FP32 and quant models.")


if __name__ == "__main__":
    fp32 = "/home/omni/Programming/QRID/QRID/models/onnx/yolov8l/fp32/yolov8l_processed.onnx"
    quant = "/home/omni/Programming/QRID/QRID/models/onnx/yolov8l/static_QInt8_QInt8/yolov8l_static_QInt8_QInt8.onnx"
    use_providers = ["CPUExecutionProvider"]
    main(fp32, quant, use_providers)
