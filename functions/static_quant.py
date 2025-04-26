import os
import cv2 # Using OpenCV for image loading/preprocessing
import numpy as np
from pathlib import Path
import onnxruntime as ort
from functions.constants import default_onnx_static_quant_kwargs
from onnxruntime.quantization import quantize_static, CalibrationDataReader

class YOLOCalibrationDataReader(CalibrationDataReader):
    def __init__(self, image_folder_or_list, model_path, batch_size=1, imgsz=640):
        self.batch_size = batch_size
        self.imgsz = imgsz # Keep imgsz as the target size for preprocessing

        # --- discover model IO automatically (from new class) ---
        sess = ort.InferenceSession(model_path, providers=['CPUExecutionProvider'])
        self.input_name = sess.get_inputs()[0].name
        # We might not strictly need h_in, w_in if using fixed imgsz preprocessing
        # _, _, self.h_in, self.w_in = sess.get_inputs()[0].shape

        # --- collect image paths (from new class) ---
        if isinstance(image_folder_or_list, (str, Path)):
            exts = (".jpg", ".jpeg", ".png", ".bmp")
            self.image_files = sorted(
                str(p) for p in Path(image_folder_or_list).iterdir()
                if p.suffix.lower() in exts and p.is_file())
        elif isinstance(image_folder_or_list, (list, tuple)):
            self.image_files = list(image_folder_or_list)
        else:
            raise TypeError("image_folder_or_list must be path or list")

        self.num_images = len(self.image_files)
        self.current_index = 0
        print(f"Initialized CalibrationDataReader with {self.num_images} images. Input name: '{self.input_name}', Target size: {self.imgsz}x{self.imgsz}")

    # --- Iterator helpers (from new class) ---
    def __len__(self):
        return self.num_images

    def rewind(self):
        self.current_index = 0
        print("Rewinding CalibrationDataReader.")
        return None

    # --- get_next with correct batching (from first example) ---
    def get_next(self):
        """
        Returns the next batch of preprocessed data. Returns None when iteration is finished.
        """
        if self.current_index >= self.num_images:
            # All data enumerated. Return None to signal completion.
            return None

        # Determine the end index for the current batch
        end_index = min(self.current_index + self.batch_size, self.num_images)
        batch_image_paths = self.image_files[self.current_index:end_index]

        # Preprocess images and collect them in a batch
        batch_data = []
        for image_path in batch_image_paths:
            try:
                processed_image = self._preprocess_image(image_path)
                batch_data.append(processed_image)
            except Exception as e:
                print(f"Warning: Skipping image {image_path} due to error: {e}")
                # Optionally adjust end_index or handle differently if an image fails

        # If all images in the intended batch failed, return None or handle appropriately
        if not batch_data:
             print(f"Warning: No images could be processed in batch starting at index {self.current_index}")
             self.current_index = end_index # Move index forward anyway
             # Decide whether to try the next batch or stop
             if self.current_index >= self.num_images:
                 return None
             else:
                 # Try fetching the next batch recursively or just continue the loop
                 # For simplicity, we'll just let it potentially return an empty dict if next call also fails
                 # A more robust solution might be needed depending on error frequency
                 pass # Let the next call handle the subsequent batch


        # Stack images into a single batch array only if batch_data is not empty
        if batch_data:
            batch_data_np = np.stack(batch_data, axis=0)
            # Update the current index for the next call
            self.current_index = end_index
            # Return data in the required dictionary format
            return {self.input_name: batch_data_np}
        else:
            # If batch is empty after skipping errors, signal end or try next
             self.current_index = end_index # Ensure index moves forward
             if self.current_index >= self.num_images:
                  return None
             else: # Attempt to get the next batch in the next call
                  return self.get_next() # Or handle as appropriate


    # --- Preprocessing function (using logic from new class for interpolation) ---
    def _preprocess_image(self, path: str) -> np.ndarray:
        """
        Loads and preprocesses a single image.
        """
        img = cv2.imread(path, cv2.IMREAD_COLOR)
        if img is None:
            raise IOError(f"Could not read {path}")
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        h0, w0 = img.shape[:2]
        # Use self.imgsz for target dimensions
        target_h, target_w = self.imgsz, self.imgsz
        r = min(target_h / h0, target_w / w0)
        nh, nw = int(round(h0 * r)), int(round(w0 * r))

        # Resize using adaptive interpolation
        if r < 1: # Downsampling
            interpolation = cv2.INTER_AREA
        else: # Upsampling or same size
            interpolation = cv2.INTER_LINEAR

        # Check if resizing is actually needed
        if (nw, nh) != (w0, h0):
            img = cv2.resize(img, (nw, nh), interpolation=interpolation)

        # Padding
        top = (target_h - nh) // 2
        bottom = target_h - nh - top
        left = (target_w - nw) // 2
        right = target_w - nw - left
        img = cv2.copyMakeBorder(img, top, bottom, left, right,
                                 cv2.BORDER_CONSTANT, value=(114, 114, 114))

        # Transpose, normalize, ensure contiguous
        img = img.transpose(2, 0, 1) # HWC -> CHW
        img = np.ascontiguousarray(img, dtype=np.float32) / 255.0 # Ensure contiguous and normalize

        return img

def static_quantization(model_input:str|Path, 
                 model_output:str|Path,
                 calibration_data: CalibrationDataReader,
                 quant_kwargs: dict = default_onnx_static_quant_kwargs):
    
    # Merge default quantization arguments with user-provided arguments
    quant_kwargs = {**default_onnx_static_quant_kwargs, **quant_kwargs}
    print(f"Performing static quantization on model {model_input}...")
    try:
        quantize_static(
            model_input=model_input,
            model_output=model_output,
            calibration_data_reader=calibration_data,
            **quant_kwargs,
        )
        print(f"Model {model_input} quantized successfully.")
    except Exception as e:
        print(f"Error during static quantization of model {model_input}: {e}. Please check the model path and quantization arguments.")
        return