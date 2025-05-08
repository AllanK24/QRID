# Analyzing Quantization Robustness to Input Degradations

## Abstract

Post-Training Quantization (PTQ) is crucial for deploying deep learning models like object detectors (e.g., YOLO) on resource-constrained edge devices. It significantly reduces model size and improves inference speed by converting model weights and activations from floating-point (FP32) to lower-precision integers (typically INT8). However, this conversion can introduce approximation errors, potentially making quantized models more sensitive or "brittle" to real-world input variations and degradations (e.g., noise, blur, low contrast, compression artifacts) compared to their FP32 counterparts. This project investigates this robustness gap. We compare the performance of a pre-trained YOLO object detector in FP32 against its quantized versions (FP16, dynamic QUInt8 ONNX, static INT8 TensorRT) when evaluated on clean data versus data subjected to common input degradations. Furthermore, we propose and evaluate a novel **Degradation-Aware Calibration** strategy for static INT8 PTQ, aiming to improve the robustness of the quantized model.

## Core Concept: Degradation-Aware Calibration (Novel Contribution)

Standard Static Post-Training Quantization typically relies on a small set of *clean* calibration data to determine the optimal quantization parameters (scale and zero-point) for model activations. While effective for clean inputs, we hypothesize that this makes the resulting INT8 model less robust to real-world image degradations, as the calibration process never observes the activation statistics produced under such conditions.

**Our Novel Approach:** We propose **Degradation-Aware Calibration**. Instead of using only clean data, we calibrate the static INT8 quantization process using a *mixed dataset*. This dataset contains both:
1.  A representative sample of clean images.
2.  Versions of images from the sample set augmented with realistic degradations, including:
    *   Gaussian Noise (Low & Medium Variance)
    *   Gaussian Blur (Low & Medium Kernel Size)
    *   Low Contrast
    *   JPEG Compression Artifacts (Moderate & Heavy)

**Hypothesis:** By exposing the calibration process to the activation statistics generated from both clean and degraded inputs, the resulting quantization parameters should be more representative of the wider range of expected runtime conditions. This is expected to lead to improved robustness, characterized by a smaller drop in accuracy (mAP) when the INT8 model is evaluated on degraded datasets, compared to an INT8 model calibrated using only clean data.

## Methodology

1.  **Base Model:** YOLO12 family of models (nano, small, medium, large, extra-large),  pre-trained on COCO.
2.  **Target Precisions & Frameworks:**
    *   **FP32 Baseline:** Exported to TensorRT (`.engine`) format for optimized inference.
    *   **FP16 Baseline:** Exported to TensorRT (`.engine`) format.
    *   **Dynamic Quantization (ONNX):** Exported to ONNX, applied pre-processing, then applied dynamic quantization using `onnxruntime.quantization.quantize_dynamic`. **Weights were quantized to QUInt8**. Activations are dynamically quantized at runtime (typically to INT8/UINT8 by ORT).
    *   **Static Quantization (TensorRT INT8):** Exported directly to INT8 TensorRT (`.engine`) using `model.export()` with `int8=True` via Ultralytics' API. This utilizes TensorRT's internal PTQ mechanism. Two versions were created based on the calibration data provided via the `data` argument:
        *   **Standard Static:** Calibrated using a set containing ~1000 *clean* images randomly sampled from the COCO train set, with guaranteed coverage of all classes.
        *   **Degradation-Aware Static (Novel):** Calibrated using a dataset a set containing ~1000 images – approximately 50% clean images and 50% images randomly degraded using the augmentations listed in the Core Concept section (noise, blur, contrast, jpeg), sourced from the clean sample pool.
3.  **Validation Datasets:**
    *   **Clean:** Standard COCO `val2017` split.
    *   **Degraded:** Separate versions of the *entire* COCO `val2017` split were created, with each version having *one* specific degradation applied uniformly to all images (Low Noise, Medium Noise, Low Blur, Medium Blur, Low Contrast, Heavy JPEG), besides that the mixed augmented validation set was created that randomly applied one of augmentations listed to 50% of images while other 50% remained clean. Corresponding `data_*.yaml` files were generated, using functionality provided in the repository.
4.  **Evaluation:**
    *   All models (FP32 engine, FP16 engine, Dynamic QUInt8 ONNX, Static INT8 Clean Calib engine, Static INT8 Mixed Calib engine) were benchmarked on the clean validation set and all distinct degraded validation sets.
    *   **Metrics:** mAP50-95(B), mAP50(B) for accuracy; Latency (ms/image) and FPS for performance (Batch Size 1).
    *   **Hardware:** NVIDIA RTX 2070

## Results Summary

This study evaluated the robustness of different YOLO model sizes (YOLOv12n, s, m, l, x - *[Adjust model name/version if needed]*) in various precisions (FP32, FP16, Dynamic QUInt8 ONNX, Static INT8 TensorRT) against common input image degradations. Static INT8 models were generated using TensorRT's PTQ, calibrated either with clean data (`Static INT8 (Clean Calib)`) or a 50/50 mix of clean and degraded data (`Static INT8 (Mixed Calib)`). Performance was measured by the relative drop in mAP50-95 and mAP50 compared to each model's own performance on the clean COCO validation set.

*(Note: Detailed tables and raw results can be found in the `results_tables/` directory.)*

**Key Findings:**

1.  **Baseline Robustness (FP32/FP16/Dynamic INT8):**
    *   FP32, FP16, and Dynamic INT8 models exhibited very similar robustness characteristics across all tested degradations and model sizes. Their relative performance drops were nearly identical for each condition.
    *   All models showed high resilience to **Low Contrast** and **Heavy JPEG Compression**, with minimal mAP drops (typically < 2%).
    *   **Blur** caused moderate performance degradation, increasing with severity (Low Blur: ~5-6% drop; Medium Blur: ~11-13% drop), consistently across models.
    *   **Gaussian Noise** proved challenging. Even "Low Noise" caused significant drops (e.g., ~11-24% for larger models, potentially higher for yolov12n if noise results were verified), and "Medium Noise" caused substantial drops (e.g., ~26-59%). This suggests the noise levels might be high relative to the models' capabilities, or noise is a particularly difficult degradation for these architectures. *(Self-Correction: The previously suspected extreme noise issue with yolov12n seems less anomalous now, given the significant drops also seen in larger models, although the magnitude still warrants verification).*
    *   Performance on the **Mixed Degradation** validation set showed moderate drops (~4-8%), reflecting an average impact of the various degradations present.

2.  **Static INT8 Quantization Impact:**
    *   **Clean Accuracy:** As expected, static INT8 TensorRT quantization incurred an initial accuracy drop on the clean validation set compared to FP32/FP16 (absolute values not shown here, but implied by the relative drop analysis).
    *   **Robustness vs FP32 (Relative Drop):**
        *   **Blur & Contrast:** Static INT8 models (both calibration methods) were often **more robust** (smaller relative mAP drop) or similarly robust to blur and low contrast compared to FP32/FP16/Dynamic INT8, especially for `yolo12s` and `yolo12m`. For `yolo12l` and `yolo12x`, the results were more mixed, with mixed calibration sometimes showing significant improvement (e.g., `yolo12x` on Low Blur, Contrast).
        *   **JPEG Heavy:** Static INT8 models showed similar or slightly higher relative drops compared to FP32, except for `yolo12x` (Mixed Calib) which showed an apparent *gain* (negative drop), potentially indicating noise or specific interactions.
        *   **Noise:** Static INT8 models consistently demonstrated **lower robustness** (larger relative mAP drop) to both low and medium noise compared to FP32/FP16/Dynamic INT8 across most model sizes. This suggests quantization exacerbates the model's sensitivity to noise.
        *   **Mixed Degradation:** The relative drop on the mixed set was generally similar or slightly higher for static INT8 models compared to FP32/FP16/Dynamic INT8, except again for `yolo12x` (Mixed Calib) which performed closer to FP32.

3.  **Degradation-Aware Calibration (Clean vs. Mixed):**
    *   **Minor Differences:** For most model sizes (`n`, `s`, `m`, `l`) and most degradations, calibrating with **mixed data showed negligible difference** in robustness compared to calibrating with clean data. The relative mAP drops were nearly identical.
    *   **Potential Effect on Larger Models:** For `yolo12m` (Noise), `yolo12l` (Contrast, JPEG, Noise Low), and particularly `yolo12x` (Blur Low, Contrast, JPEG, Noise Low/Medium, Mixed), **mixed calibration demonstrated potentially different robustness** compared to clean calibration.
        *   On `yolo12x`, mixed calibration led to significantly *better* robustness (smaller mAP drop) for Low Blur, Contrast, JPEG, and Low/Medium Noise compared to clean calibration INT8.
        *   However, for some cases like `yolo12x` Medium Blur and the overall Mixed Degradation set, clean calibration INT8 performed better (smaller drop) than mixed calibration INT8.
    *   **Conclusion on Novelty:** The proposed Degradation-Aware Calibration strategy using a 50/50 mix did not yield consistent robustness improvements across all model sizes and degradations. While it showed potentially significant benefits for the largest model (`yolo12x`) under several conditions, it offered little advantage for smaller models and sometimes performed slightly worse than standard clean calibration. The effectiveness appears highly dependent on model size and potentially the specific degradation type.

**Overall:** Static INT8 TensorRT quantization offers significant speed benefits but comes with a baseline accuracy cost and increased sensitivity to noise compared to FP32/FP16/Dynamic INT8. While static INT8 can sometimes be slightly more robust to blur or contrast changes, the effectiveness of using mixed calibration data to further enhance general robustness was inconsistent across model scales in this study, showing notable potential only for the largest (`yolo12x`) model under specific degradations.

<!--## Repository Structure-->

## 📜 License

This project is licensed under the [MIT License](LICENSE).  
You are free to use, modify, and distribute this code with proper attribution.

## 📖 Citation

If you use this codebase or findings in your own research, please consider citing:

Kazakov, A., & Karimov, T. (2025). *Analyzing Quantization Robustness to Input Degradations*. Master's Thesis, Bahcesehir University.

BibTeX:
```bibtex
@mastersthesis{kazakov2025quantization,
  title={Analyzing Quantization Robustness to Input Degradations},
  author={Kazakov, Allan and Karimov, Toghrul},
  school={Bahcesehir University},
  year={2025}
}