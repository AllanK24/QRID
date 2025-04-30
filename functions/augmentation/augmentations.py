import os
import json
import random
import cv2 # Make sure OpenCV is installed: pip install opencv-python
import numpy as np
import albumentations as A
from pathlib import Path
from tqdm.auto import tqdm # For progress bars: pip install tqdm
import shutil # Import shutil for file copying

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
transform_jpeg_mod = A.ImageCompression(quality_lower=jpeg_quality_mod[0], quality_upper=jpeg_quality_mod[1], compression_type="jpeg", p=1.0)
transform_jpeg_heavy = A.ImageCompression(quality_lower=jpeg_quality_heavy[0], quality_upper=jpeg_quality_heavy[1], compression_type="jpeg", p=1.0)


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


def create_mixed_calibration_set( # Renamed function to avoid conflict
    clean_image_paths: list[str],
    output_dir: str,
    target_total_size: int,
    degradation_ratio: float = 0.5,
    seed: int = 42,
):
    """
    Creates a mixed calibration set by applying degradations to a subset of
    clean images and copying another subset of clean images, saving all
    results directly into the output directory while preserving original filenames.

    Args:
        clean_image_paths (list[str]): List of full paths to the original clean calibration images.
        output_dir (str): Path to the directory where the final mixed set
                          (degraded + copied clean images) and the JSON list will be saved.
                          Existing files with the same name will be overwritten.
        target_total_size (int): The desired total number of images in the final mixed list.
                                 The actual size may be smaller if the number of unique
                                 source images is less than this target.
        degradation_ratio (float): The target proportion of degraded images in the final set (0.0 to 1.0).
        seed (int): Random seed for reproducibility.

    Returns:
        list[str] | None: A list of the full paths to the mixed calibration images
                          *within the output_dir*, or None if an error occurs.
    """
    random.seed(seed)
    np.random.seed(seed) # For albumentations randomness

    if not (0.0 <= degradation_ratio <= 1.0):
        print("Error: degradation_ratio must be between 0.0 and 1.0")
        return None

    output_path = Path(output_dir)

    try:
        # Create output directory if it doesn't exist
        output_path.mkdir(parents=True, exist_ok=True)
        print(f"Output directory: {output_path} (Clean and degraded images will be saved here)")
    except Exception as e:
        print(f"Error creating output directory: {e}")
        return None

    num_total_clean_available = len(clean_image_paths)
    if num_total_clean_available == 0:
        print("Error: Input clean_image_paths list is empty.")
        return None

    # --- Calculate Image Counts ---
    # The total number of images we can possibly use is limited by the available clean images,
    # as each source image will be used at most once (either degraded or copied clean).
    effective_target_size = min(target_total_size, num_total_clean_available)
    if effective_target_size < target_total_size:
        print(f"Warning: Target size ({target_total_size}) is larger than available unique "
              f"source images ({num_total_clean_available}). Final set size will be limited to {effective_target_size}.")

    num_actual_degraded = int(round(effective_target_size * degradation_ratio))
    num_actual_clean_to_copy = effective_target_size - num_actual_degraded

    # Ensure counts don't exceed available images (though effective_target_size should handle this)
    num_actual_degraded = min(num_actual_degraded, num_total_clean_available)
    num_actual_clean_to_copy = min(num_actual_clean_to_copy, num_total_clean_available - num_actual_degraded)

    print(f"Target total size: {target_total_size}")
    print(f"Effective target size (limited by source images): {effective_target_size}")
    print(f"Target degradation ratio: {degradation_ratio:.2f}")
    print(f"Number of images to degrade: {num_actual_degraded}")
    print(f"Number of clean images to copy: {num_actual_clean_to_copy}")
    print(f"Total images to process: {num_actual_degraded + num_actual_clean_to_copy}")

    if num_actual_degraded + num_actual_clean_to_copy == 0:
        print("Error: Calculated number of images to process is zero.")
        return None

    # --- Select Images ---
    # Randomly select indices for degradation and copying
    indices = list(range(num_total_clean_available))
    random.shuffle(indices)

    indices_to_degrade = set(indices[:num_actual_degraded])
    indices_to_copy_clean = set(indices[num_actual_degraded : num_actual_degraded + num_actual_clean_to_copy])

    paths_to_degrade = [clean_image_paths[i] for i in indices_to_degrade]
    paths_to_copy_clean = [clean_image_paths[i] for i in indices_to_copy_clean]

    final_output_paths = []

    # --- Generate and Save Degraded Images ---
    if paths_to_degrade:
        print(f"\nApplying degradations to {len(paths_to_degrade)} images...")
        for clean_path_str in tqdm(paths_to_degrade, desc="Degrading Images"):
            clean_path = Path(clean_path_str)
            # Use original filename for output
            output_filename = clean_path.name
            output_path_img = output_path / output_filename

            try:
                # Load image using OpenCV
                img = cv2.imread(clean_path_str)
                if img is None:
                    print(f"Warning: Could not read image {clean_path_str}. Skipping.")
                    continue

                # Apply one of the chosen degradations
                augmented_data = apply_one_degradation(image=img)
                degraded_img_np = augmented_data['image']

                # Save the degraded image (overwrites if exists)
                success = cv2.imwrite(str(output_path_img), degraded_img_np)
                if success:
                    final_output_paths.append(str(output_path_img))
                else:
                    print(f"Warning: Failed to save degraded image {output_path_img}. Skipping.")

            except Exception as e:
                print(f"Error processing or saving degraded version of {clean_path_str} to {output_path_img}: {e}. Skipping.")
        print(f"Finished degrading images.")

    # --- Copy Selected Clean Images ---
    if paths_to_copy_clean:
        print(f"\nCopying {len(paths_to_copy_clean)} clean images...")
        for clean_path_str in tqdm(paths_to_copy_clean, desc="Copying Clean Images"):
            source_path = Path(clean_path_str)
            # Use original filename for output
            dest_filename = source_path.name
            dest_path = output_path / dest_filename

            try:
                # Copy the clean image (overwrites if exists)
                # shutil.copy2 preserves metadata like modification time
                shutil.copy2(str(source_path), str(dest_path))
                final_output_paths.append(str(dest_path))
            except Exception as e:
                print(f"Error copying clean image {source_path} to {dest_path}: {e}. Skipping.")
        print(f"Finished copying clean images.")


    # --- Combine and Save Final List ---
    random.shuffle(final_output_paths) # Shuffle the final list

    final_count = len(final_output_paths)
    print(f"\nTotal images in the final mixed set ({output_dir}): {final_count}")

    # Verify count matches expected (adjusting for potential read/write/copy errors)
    expected_count = len(paths_to_degrade) + len(paths_to_copy_clean)
    if final_count != expected_count:
         print(f"Warning: Expected {expected_count} images based on selection, but generated/copied {final_count}. "
               "Some files might have failed processing or saving.")


    output_json_path = output_path / "mixed_calibration_paths.json"
    try:
        print(f"Saving final list of {final_count} paths (relative to output dir) to {output_json_path}...")
        # Store the absolute paths in the JSON for clarity, although relative could also work
        with open(output_json_path, 'w') as f:
            json.dump(final_output_paths, f, indent=4)
        print(f"Successfully saved final path list to {output_json_path}")
        return final_output_paths
    except Exception as e:
        print(f"Error saving final JSON list: {e}")
        return None


# --- Example Usage ---
if __name__ == "__main__":
    # Assume you have the list of clean paths from the previous sampling step
    image_dir = Path("/home/omni/Programming/QRID/datasets/coco/images/val2017")
    val_path = [str(p) for p in image_dir.iterdir() if p.is_file()]
    ratios = [
        0.25,
        0.5,
        0.75,
    ]
    for ratio in ratios:
        print(f"\nGenerating mixed calibration set with degradation ratio: {ratio}")
        
        output_directory = f"/home/omni/Programming/QRID/QRID/augmented_sets/calibration_data_val_2017_{ratio}" # Base directory for outputs
        target_size = 5000 # Total size of the mixed set
        try:
            # Generate the mixed set
            mixed_paths = create_mixed_calibration_set(
                clean_image_paths=val_path,
                output_dir=output_directory,
                target_total_size=target_size,
                degradation_ratio=ratio,
                seed=42
            )
            print(f"Mixed calibration set created with {len(mixed_paths)} images at {output_directory}. Ratio: {ratio}")
        except Exception as e:
            print(f"An error occurred: {e}")