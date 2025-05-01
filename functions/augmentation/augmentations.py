import os
import json
import random
import cv2 # Make sure OpenCV is installed: pip install opencv-python
import numpy as np
import albumentations as A
from pathlib import Path
from tqdm.auto import tqdm # For progress bars: pip install tqdm
import shutil # Import for file copying

# --- Augmentation Definitions (Same as before) ---
noise_var_low = (10.0, 50.0)
noise_var_med = (50.0, 150.0)
blur_kernel_low = (3, 5)
blur_kernel_med = (7, 11)
contrast_limit_low = (-0.6, -0.3) # Reduce contrast
jpeg_quality_mod = (50, 75)
jpeg_quality_heavy = (20, 45)
transform_noise_low = A.GaussNoise(var_limit=noise_var_low, p=1.0)
transform_noise_med = A.GaussNoise(var_limit=noise_var_med, p=1.0)
transform_blur_low = A.GaussianBlur(blur_limit=blur_kernel_low, p=1.0)
transform_blur_med = A.GaussianBlur(blur_limit=blur_kernel_med, p=1.0)
transform_contrast_low = A.RandomBrightnessContrast(brightness_limit=0, contrast_limit=contrast_limit_low, p=1.0)
transform_jpeg_mod = A.ImageCompression(quality_lower=jpeg_quality_mod[0], quality_upper=jpeg_quality_mod[1], compression_type="jpeg", p=1.0)
transform_jpeg_heavy = A.ImageCompression(quality_lower=jpeg_quality_heavy[0], quality_upper=jpeg_quality_heavy[1], compression_type="jpeg", p=1.0)
apply_one_degradation = A.OneOf([
    transform_noise_low,
    transform_noise_med,
    transform_blur_low,
    transform_blur_med,
    transform_contrast_low,
    transform_jpeg_mod,
    transform_jpeg_heavy,
], p=1.0)
# --- End Augmentation Definitions ---


def create_mixed_calibration_set( # Renamed again
    clean_image_paths: list[str],
    output_dir: str,
    target_total_size: int,
    degradation_ratio: float = 0.5,
    seed: int = 42,
):
    """
    Creates a mixed calibration set by selecting a subset of clean images,
    degrading a portion of them, and saving ALL images (original or degraded)
    into the output directory using their ORIGINAL filenames.

    Args:
        clean_image_paths (list[str]): List of full paths to the original clean calibration images.
        output_dir (str): Path to the directory where selected images (either copied
                          clean or generated degraded) and the final JSON list will be saved.
                          All images will retain their original names.
        target_total_size (int): The desired total number of images in the final mixed list/folder.
        degradation_ratio (float): The target proportion of degraded images within the selected set (0.0 to 1.0).
        seed (int): Random seed for reproducibility.

    Returns:
        list[str] | None: A list of the full paths to the mixed calibration images
                          saved in the output_dir (using original names), or None if an error occurs.
    """
    random.seed(seed)
    np.random.seed(seed) # For albumentations randomness

    if not (0.0 <= degradation_ratio <= 1.0):
        print("Error: degradation_ratio must be between 0.0 and 1.0")
        return None

    output_path = Path(output_dir)

    try:
        # Create the output directory if it doesn't exist
        output_path.mkdir(parents=True, exist_ok=True)
        print(f"Output directory (for images and JSON): {output_path}")
    except Exception as e:
        print(f"Error creating output directory: {e}")
        return None

    num_total_clean = len(clean_image_paths)
    if num_total_clean == 0:
        print("Error: Input clean_image_paths list is empty.")
        return None

    # --- Select the subset of images that will be in the final output ---
    actual_set_size = min(target_total_size, num_total_clean)
    if actual_set_size < target_total_size:
        print(f"Warning: Target size {target_total_size} is larger than available "
              f"clean images ({num_total_clean}). Using {actual_set_size} images.")
    if actual_set_size == 0:
         print("Error: Target size or available images is zero. Cannot proceed.")
         return None

    print(f"Selecting {actual_set_size} images for the final set.")
    indices = list(range(num_total_clean))
    random.shuffle(indices)
    selected_indices = indices[:actual_set_size]
    selected_original_paths = [clean_image_paths[i] for i in selected_indices]

    # --- Decide which of the SELECTED images to degrade ---
    num_to_degrade = int(round(actual_set_size * degradation_ratio))
    indices_within_selected = list(range(actual_set_size))
    random.shuffle(indices_within_selected)
    indices_to_degrade_in_selected = set(indices_within_selected[:num_to_degrade])

    num_to_copy_clean = actual_set_size - num_to_degrade

    print(f"Target total size: {target_total_size} (Actual: {actual_set_size})")
    print(f"Target degradation ratio within selected: {degradation_ratio:.2f}")
    print(f"Number of images to degrade: {num_to_degrade}")
    print(f"Number of clean images to copy: {num_to_copy_clean}")

    # --- Process each selected image: either copy or degrade+save ---
    final_output_paths = []
    print(f"\nProcessing {actual_set_size} selected images (copying or degrading)...")

    for i, original_path_str in enumerate(tqdm(selected_original_paths, desc="Processing Images")):
        original_path = Path(original_path_str)
        # Destination path uses the ORIGINAL filename inside the output directory
        destination_path = output_path / original_path.name

        # Check if the index 'i' within the selected list is marked for degradation
        if i in indices_to_degrade_in_selected:
            # Degrade and Save
            try:
                img = cv2.imread(original_path_str)
                if img is None:
                    print(f"Warning: Could not read image {original_path_str}. Skipping degradation.")
                    continue

                # Apply one degradation
                augmented_data = apply_one_degradation(image=img)
                degraded_img_np = augmented_data['image']

                # Save degraded image with original name
                success = cv2.imwrite(str(destination_path), degraded_img_np)
                if success:
                    final_output_paths.append(str(destination_path))
                else:
                    print(f"Warning: Failed to save degraded image {destination_path}. Skipping.")

            except Exception as e:
                print(f"Error processing or saving degraded version of {original_path_str} as {destination_path}: {e}. Skipping.")
        else:
            # Copy Clean
            try:
                # Use copy2 to preserve metadata if possible
                shutil.copy2(original_path_str, destination_path)
                final_output_paths.append(str(destination_path))
            except FileNotFoundError:
                 print(f"Warning: Source file not found, cannot copy: {original_path_str}. Skipping.")
            except Exception as e:
                print(f"Warning: Error copying {original_path_str} to {destination_path}: {e}. Skipping.")

    # --- Save Final List ---
    final_count = len(final_output_paths)
    print(f"\nSuccessfully processed and saved {final_count} images to {output_path}")
    if final_count != actual_set_size:
         print(f"Warning: Expected {actual_set_size} images in final set, but processed {final_count}. Check logs for skipped files.")

    # Shuffle the final list of paths within the output directory
    random.shuffle(final_output_paths)

    output_json_path = output_path / "mixed_calibration_paths.json"
    try:
        print(f"Saving final list of {final_count} paths (all within {output_path} with original names) to {output_json_path}...")
        with open(output_json_path, 'w') as f:
            json.dump(final_output_paths, f, indent=4)
        print("Successfully saved final path list.")
        return final_output_paths
    except Exception as e:
        print(f"Error saving final JSON list: {e}")
        return None

# --- Example Usage ---
if __name__ == "__main__":
    # Assume you have the list of clean paths from the previous sampling step
    with open("/home/omni/Programming/QRID/QRID/imgs_calibrated_for_ptq/calibration_files_sampled_final.json", "r") as f:
        calibration_image_paths = json.load(f)
    
    ratios = [
        0.25,
        0.5,
        0.75,
    ]
    
    for ratio in ratios:
        print(f"\nGenerating mixed calibration set with degradation ratio: {ratio}")
        
        output_directory = f"/home/omni/Programming/QRID/QRID/augmented_sets/calibration_data_from_train2017_{ratio}" # Base directory for outputs
        target_size = 1000 # Total size of the mixed set
        try:
            print("Applying augmentations and generating mixed calibration set for ratio:", ratio)
            
            # Generate the mixed set
            mixed_paths = create_mixed_calibration_set(
                clean_image_paths=calibration_image_paths,
                output_dir=output_directory,
                target_total_size=target_size,
                degradation_ratio=ratio,
                seed=42
            )

            print(f"Mixed calibration set generated with {len(mixed_paths)} images. Ratio: {ratio}")

        except Exception as e:
            print(f"An error occurred: {e}")