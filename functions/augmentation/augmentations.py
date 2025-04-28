import os
import json
import random
import cv2 # Make sure OpenCV is installed: pip install opencv-python
import numpy as np
import albumentations as A
from pathlib import Path
from tqdm.auto import tqdm # For progress bars: pip install tqdm

# --- Augmentation Definitions (Copied from your input) ---
# Define ranges for different levels of degradation
noise_var_low = (10.0, 50.0)
noise_var_med = (50.0, 150.0)
blur_kernel_low = (3, 5)
blur_kernel_med = (7, 11)
contrast_limit_low = (-0.6, -0.3) # Reduce contrast
jpeg_quality_mod = (50, 75)
jpeg_quality_heavy = (20, 45)

# Create individual transforms
transform_noise_low = A.GaussNoise(var_limit=noise_var_low, p=1.0)
transform_noise_med = A.GaussNoise(var_limit=noise_var_med, p=1.0)
transform_blur_low = A.GaussianBlur(blur_limit=blur_kernel_low, p=1.0)
transform_blur_med = A.GaussianBlur(blur_limit=blur_kernel_med, p=1.0)
transform_contrast_low = A.RandomBrightnessContrast(brightness_limit=0, contrast_limit=contrast_limit_low, p=1.0)
# Corrected Albumentations transform for JPEG
transform_jpeg_mod = A.ImageCompression(quality_lower=jpeg_quality_mod[0], quality_upper=jpeg_quality_mod[1], compression_type=A.ImageCompression.ImageCompressionType.JPEG, p=1.0)
transform_jpeg_heavy = A.ImageCompression(quality_lower=jpeg_quality_heavy[0], quality_upper=jpeg_quality_heavy[1], compression_type=A.ImageCompression.ImageCompressionType.JPEG, p=1.0)


# Combine using OneOf to apply only ONE degradation per image randomly
# p=1.0 ensures one of these is always chosen for the degraded images
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


def create_mixed_calibration_set(
    clean_image_paths: list[str],
    output_dir: str,
    target_total_size: int,
    degradation_ratio: float = 0.5,
    seed: int = 42,
):
    """
    Creates a mixed calibration set by applying degradations to a subset of clean images.

    Args:
        clean_image_paths (list[str]): List of full paths to the original clean calibration images.
        output_dir (str): Path to the directory where degraded images and the final JSON list will be saved.
        target_total_size (int): The desired total number of images in the final mixed list.
        degradation_ratio (float): The target proportion of degraded images in the final set (0.0 to 1.0).
        seed (int): Random seed for reproducibility.

    Returns:
        list[str] | None: A list of the full paths to the mixed (clean + degraded)
                          calibration images, or None if an error occurs.
    """
    random.seed(seed)
    np.random.seed(seed) # For albumentations randomness

    if not (0.0 <= degradation_ratio <= 1.0):
        print("Error: degradation_ratio must be between 0.0 and 1.0")
        return None

    output_path = Path(output_dir)
    degraded_images_subdir = output_path / "degraded_calibration_images"

    try:
        # Create directories if they don't exist
        output_path.mkdir(parents=True, exist_ok=True)
        degraded_images_subdir.mkdir(parents=True, exist_ok=True)
        print(f"Output directory: {output_path}")
        print(f"Degraded images will be saved to: {degraded_images_subdir}")
    except Exception as e:
        print(f"Error creating output directories: {e}")
        return None

    num_total_clean = len(clean_image_paths)
    if num_total_clean == 0:
        print("Error: Input clean_image_paths list is empty.")
        return None

    # Calculate how many images to degrade and how many to keep clean
    # Ensure we don't try to degrade more images than available
    num_target_degraded = int(round(target_total_size * degradation_ratio))
    num_to_degrade = min(num_target_degraded, num_total_clean)

    num_target_clean = target_total_size - num_to_degrade # Keep enough clean ones to reach target
    num_to_keep_clean = min(num_target_clean, num_total_clean - num_to_degrade) # Don't select from those chosen for degradation

    print(f"Target total size: {target_total_size}")
    print(f"Target degradation ratio: {degradation_ratio:.2f}")
    print(f"Number of images to degrade: {num_to_degrade} (out of {num_total_clean} available clean)")
    print(f"Number of clean images to keep: {num_to_keep_clean}")

    if num_to_degrade + num_to_keep_clean > num_total_clean:
         print(f"Warning: Cannot select {num_to_degrade} for degradation and {num_to_keep_clean} to keep clean "
               f"from only {num_total_clean} source images. Adjusting...")
         # Prioritize degradation count if possible, then fill with remaining clean
         num_to_degrade = min(num_target_degraded, num_total_clean)
         num_to_keep_clean = min(num_total_clean - num_to_degrade, num_target_clean)
         print(f"Adjusted: Degrading {num_to_degrade}, Keeping {num_to_keep_clean}")

    if num_to_degrade + num_to_keep_clean == 0:
        print("Error: Calculated number of images to process is zero.")
        return None

    # Randomly select images to degrade, the rest will be considered for keeping clean
    indices = list(range(num_total_clean))
    random.shuffle(indices)
    indices_to_degrade = set(indices[:num_to_degrade])
    indices_available_clean = indices[num_to_degrade:]

    paths_to_degrade = [clean_image_paths[i] for i in indices_to_degrade]
    # Select the required number of clean paths from the remaining pool
    paths_to_keep_clean = random.sample([clean_image_paths[i] for i in indices_available_clean], num_to_keep_clean)

    # --- Generate and Save Degraded Images ---
    print(f"\nApplying degradations to {len(paths_to_degrade)} images...")
    generated_degraded_paths = []
    for clean_path_str in tqdm(paths_to_degrade, desc="Processing Images"):
        clean_path = Path(clean_path_str)
        try:
            # Load image using OpenCV (Albumentations prefers BGR numpy arrays)
            img = cv2.imread(clean_path_str)
            if img is None:
                print(f"Warning: Could not read image {clean_path_str}. Skipping.")
                continue

            # Apply one of the chosen degradations
            augmented_data = apply_one_degradation(image=img)
            degraded_img_np = augmented_data['image']

            # Construct output filename
            # Adding a random suffix to handle potential duplicates if source names are identical
            random_suffix = ''.join(random.choices("abcdefghijklmnopqrstuvwxyz0123456789", k=4))
            output_filename = f"{clean_path.stem}_degraded_{random_suffix}{clean_path.suffix}"
            output_path_img = degraded_images_subdir / output_filename

            # Save the degraded image
            success = cv2.imwrite(str(output_path_img), degraded_img_np)
            if success:
                generated_degraded_paths.append(str(output_path_img))
            else:
                print(f"Warning: Failed to save degraded image {output_path_img}. Skipping.")

        except Exception as e:
            print(f"Error processing or saving degraded version of {clean_path_str}: {e}. Skipping.")

    print(f"Successfully generated and saved {len(generated_degraded_paths)} degraded images.")

    # --- Combine and Save Final List ---
    final_mixed_paths = paths_to_keep_clean + generated_degraded_paths
    random.shuffle(final_mixed_paths) # Shuffle the final list

    final_count = len(final_mixed_paths)
    print(f"\nTotal images in final mixed set: {final_count}")
    if final_count != num_to_keep_clean + len(generated_degraded_paths):
         print("Warning: Discrepancy in final count logic.") # Should not happen

    output_json_path = output_path / "mixed_calibration_paths.json"
    try:
        print(f"Saving final list of {final_count} paths to {output_json_path}...")
        with open(output_json_path, 'w') as f:
            json.dump(final_mixed_paths, f, indent=4)
        print("Successfully saved final path list.")
        return final_mixed_paths
    except Exception as e:
        print(f"Error saving final JSON list: {e}")
        return None

# --- Example Usage ---
if __name__ == "__main__":
    # Assume you have the list of clean paths from the previous sampling step
    clean_calibration_list_file = "/home/omni/Programming/QRID/QRID/imgs_calibrated_for_ptq/calibration_files_sampled_final.json" # From previous step
    output_directory = "./mixed_calibration_data" # Base directory for outputs
    target_size = 1000
    ratio = 0.5 # 50% degraded

    print(f"Reading clean calibration paths from: {clean_calibration_list_file}")
    try:
        with open(clean_calibration_list_file, 'r') as f:
            clean_paths = json.load(f)
        print(f"Read {len(clean_paths)} clean paths.")

        if clean_paths:
            # Generate the mixed set
            mixed_paths = create_mixed_calibration_set(
                clean_image_paths=clean_paths,
                output_dir=output_directory,
                target_total_size=target_size,
                degradation_ratio=ratio,
                seed=42
            )

            if mixed_paths:
                print(f"\nMixed calibration set generation complete. Final list size: {len(mixed_paths)}")
                print(f"Degraded images saved in: {Path(output_directory) / 'degraded_calibration_images'}")
                print(f"Final path list saved to: {Path(output_directory) / 'mixed_calibration_paths.json'}")

    except FileNotFoundError:
        print(f"Error: Clean calibration list file not found at {clean_calibration_list_file}")
    except Exception as e:
        print(f"An error occurred: {e}")