import os
import cv2 # Using OpenCV for image loading/preprocessing
import numpy as np
from pathlib import Path
from functions.constants import default_onnx_static_quant_kwargs
from onnxruntime.quantization import quantize_static, CalibrationDataReader

class YOLOCalibrationDataReader(CalibrationDataReader):
    def __init__(self, image_folder_or_list, batch_size=1, input_name='images', imgsz=640):
        """
        Initializes the data reader.

        Args:
            image_folder_or_list (str or list): Path to the folder containing calibration images
                                                or a list of image file paths.
            model_path (str): Path to the ONNX model file (used to get input details).
            batch_size (int): Number of images to process in each batch.
            input_name (str): The name of the input node in the ONNX model.
            imgsz (int): The target image size (e.g., 640).
        """
        self.imgsz = imgsz
        self.batch_size = batch_size
        self.input_name = input_name

        # Get image file paths
        if isinstance(image_folder_or_list, str):
            self.image_files = [os.path.join(image_folder_or_list, f) for f in os.listdir(image_folder_or_list)
                                if os.path.isfile(os.path.join(image_folder_or_list, f))]
        elif isinstance(image_folder_or_list, list):
            self.image_files = image_folder_or_list
        else:
            raise ValueError("image_folder_or_list must be a folder path (str) or a list of file paths.")

        self.image_files.sort() # Ensure consistent order
        self.num_images = len(self.image_files)
        self.current_index = 0

        print(f"Initialized CalibrationDataReader with {self.num_images} images.")

    def get_next(self):
        """
        Returns the next batch of preprocessed data. Needs to return a dictionary
        mapping input names to numpy arrays. Returns None when iteration is finished.
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
            processed_image = self._preprocess_image(image_path)
            batch_data.append(processed_image)

        # Stack images into a single batch array
        batch_data_np = np.stack(batch_data, axis=0)

        # Update the current index for the next call
        self.current_index = end_index

        # Return data in the required dictionary format
        # Input name must match the ONNX model's input name!
        return {self.input_name: batch_data_np}

    def _preprocess_image(self, image_path):
        """
        Loads and preprocesses a single image to match YOLOv8 input requirements.
        This needs to replicate the preprocessing used during training/export.
        """
        try:
            # Load image using OpenCV
            img = cv2.imread(image_path)
            if img is None:
                raise IOError(f"Could not read image: {image_path}")

            # --- Start YOLOv8 Preprocessing ---
            # 1. Letterbox/Resize to imgsz
            img_h, img_w = img.shape[:2]
            w, h = self.imgsz, self.imgsz
            r = min(h / img_h, w / img_w) # Calculate resize ratio
            new_unpad_w, new_unpad_h = int(round(img_w * r)), int(round(img_h * r))
            dw, dh = (w - new_unpad_w) / 2, (h - new_unpad_h) / 2 # Padding

            if (img_w, img_h) != (new_unpad_w, new_unpad_h): # Resize if necessary
                img = cv2.resize(img, (new_unpad_w, new_unpad_h), interpolation=cv2.INTER_LINEAR)

            top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
            left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
            img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=(114, 114, 114)) # Add gray padding

            # 2. BGR to RGB
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            # 3. HWC to CHW
            img = img.transpose(2, 0, 1) # CHW

            # 4. Normalize to [0.0, 1.0] and convert to float32
            img = np.ascontiguousarray(img, dtype=np.float32) / 255.0

            # --- End YOLOv8 Preprocessing ---

            return img

        except Exception as e:
            print(f"Error processing image {image_path}: {e}")
            # Handle error appropriately - skip image, return dummy data, or raise
            # For simplicity here, we'll raise it, but you might want to just skip.
            raise

# Example Usage (Conceptual - integrate into your quantization script)

# Define paths and parameters
clean_calib_files = [...] # List of paths to your clean calibration images
mixed_calib_files = [...] # List of paths to your mixed (clean + degraded) calibration images
model_onnx_path = "path/to/yolov8n_processed.onnx" # Path to your preprocessed FP32 model
input_name = "images" # Check this matches your model's input name (use Netron)
img_size = 640
batch_size = 16 # Adjust based on memory

# Create Data Reader instances
print("Creating Clean Calibration Data Reader...")
clean_reader = YOLOCalibrationDataReader(clean_calib_files, model_onnx_path, batch_size=batch_size, input_name=input_name, imgsz=img_size)

print("\nCreating Mixed/Degraded Calibration Data Reader...")
degraded_reader = YOLOCalibrationDataReader(mixed_calib_files, model_onnx_path, batch_size=batch_size, input_name=input_name, imgsz=img_size)

# Now pass the appropriate reader to quantize_static:
# For standard PTQ:
# quantize_static(..., calibration_data_reader=clean_reader, ...)

# For your novel degradation-aware PTQ:
# quantize_static(..., calibration_data_reader=degraded_reader, ...)

def static_quant(model_input:str|Path, 
                 model_output:str|Path,
                 calibration_data: CalibrationDataReader,
                 static_quant_kwargs: dict = default_onnx_static_quant_kwargs):
    
    # Merge default quantization arguments with user-provided arguments
    static_quant_kwargs = {**default_onnx_static_quant_kwargs, **static_quant_kwargs}
    print(f"Performing static quantization on model {model_input}...")
    try:
        quantize_static(
            model_input=model_input,
            model_output=model_output,
            calibration_data_reader=calibration_data
            **static_quant_kwargs,
        )
        print(f"Model {model_input} quantized successfully.")
    except Exception as e:
        print(f"Error during static quantization of model {model_input}: {e}. Please check the model path and quantization arguments.")
        return