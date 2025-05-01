import os
import json
import yaml # Make sure PyYAML is installed: pip install pyyaml
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
    
    

# Assume your COCO class names are defined elsewhere or loaded
# Example: coco_class_names = [...] (List of 80 names)

def create_single_degraded_val_set(
    original_val_images_dir: str,
    original_coco_labels_dir: str, # Path to parent 'labels' dir
    output_dataset_root: str,
    degradation_name: str, # e.g., "noisy_low", "blurry_medium"
    albumentations_transform: A.Compose, # The specific transform to apply
    coco_class_names: list[str] # List of class names for data.yaml
):
    """
    Creates a degraded COCO-like validation set by applying a specific
    Albumentations transform to all images in the original validation set.

    Args:
        original_val_images_dir (str): Path to the original COCO val images (e.g., .../coco/images/val2017).
        original_coco_labels_dir (str): Path to the original COCO 'labels' directory.
        output_dataset_root (str): The root directory for the new degraded dataset structure.
        degradation_name (str): A short name for the degradation (used in paths/yaml).
        albumentations_transform (A.Compose): The specific Albumentations transform to apply to each image.
        coco_class_names (list[str]): List of COCO class names.

    Returns:
        str | None: Path to the generated data.yaml file if successful, None otherwise.
    """
    orig_img_path = Path(original_val_images_dir)
    orig_lbl_path = Path(original_coco_labels_dir)
    root_path = Path(output_dataset_root)
    images_path = root_path / "images"
    labels_path = root_path / "labels"
    val_images_path = images_path / "val2017"
    val_labels_path = labels_path / "val2017" # Need specific val labels dir
    train_images_path = images_path / "train2017" # Empty dir potentially needed

    print(f"\n--- Creating Degraded Validation Set: {degradation_name} ---")
    print(f"Output root: {root_path}")

    if not orig_img_path.is_dir():
        print(f"Error: Original validation image directory not found: {orig_img_path}")
        return None
    if not (orig_lbl_path / "val2017").is_dir(): # Check for the specific val2017 subdir
         print(f"Error: Original validation labels directory ('val2017' subdir) not found within: {orig_lbl_path}")
         return None

    try:
        # Create directories
        root_path.mkdir(parents=True, exist_ok=True)
        images_path.mkdir(exist_ok=True)
        labels_path.mkdir(exist_ok=True)
        val_images_path.mkdir(exist_ok=True)
        val_labels_path.mkdir(exist_ok=True) # Create destination for val labels
        train_images_path.mkdir(exist_ok=True) # Create empty train images dir

        # --- Copy Labels (Only val2017 needed) ---
        print(f"Copying val2017 labels from {orig_lbl_path / 'val2017'}...")
        source_val_labels = orig_lbl_path / "val2017"
        # Copy contents of source_val_labels into val_labels_path
        for item in tqdm(os.listdir(source_val_labels), desc="Copying labels"):
             s = source_val_labels / item
             d = val_labels_path / item
             if s.is_file() and s.suffix == '.txt': # Ensure it's a label file
                 shutil.copy2(s, d)

        # --- Process and Save Degraded Images ---
        print(f"Applying '{degradation_name}' degradation to images...")
        image_files = sorted([p for p in orig_img_path.glob("*") if p.suffix.lower() in (".jpg", ".jpeg", ".png", ".bmp", ".webp")])
        processed_image_paths_relative = [] # Store relative paths for val2017.txt
        processed_count = 0
        skipped_count = 0

        for img_file_path in tqdm(image_files, desc=f"Degrading ({degradation_name})"):
            try:
                img = cv2.imread(str(img_file_path))
                if img is None:
                    print(f"Warning: Could not read image {img_file_path}. Skipping.")
                    skipped_count += 1
                    continue

                # Apply the specific degradation transform
                augmented_data = albumentations_transform(image=img)
                degraded_img_np = augmented_data['image']

                # Save degraded image with the original name to the new location
                dest_img_path = val_images_path / img_file_path.name
                success = cv2.imwrite(str(dest_img_path), degraded_img_np)

                if success:
                    # Store relative path for val2017.txt
                    relative_path = dest_img_path.relative_to(root_path)
                    processed_image_paths_relative.append(f"./{relative_path}")
                    processed_count += 1
                else:
                    print(f"Warning: Failed to save degraded image {dest_img_path}. Skipping.")
                    skipped_count += 1

            except Exception as e:
                print(f"Error processing image {img_file_path}: {e}. Skipping.")
                skipped_count += 1

        print(f"Processed {processed_count} images, skipped {skipped_count}.")
        if processed_count == 0:
            print("Error: No images were successfully processed.")
            return None

        # --- Create val2017.txt ---
        val_txt_path = root_path / "val2017.txt"
        print(f"Creating {val_txt_path}...")
        try:
            with open(val_txt_path, 'w') as f_val:
                for rel_path in sorted(processed_image_paths_relative): # Sort for consistency
                    f_val.write(f"{rel_path}\n")
        except Exception as e:
             print(f"Error writing {val_txt_path}: {e}")
             return None

        # --- Create data.yaml ---
        yaml_path = root_path / f"data_{degradation_name}.yaml" # Unique yaml name
        print(f"Creating {yaml_path}...")
        num_classes = len(coco_class_names)
        yaml_data = {
            'path': str(root_path.resolve()), # Absolute path
            'train': 'train2017.txt', # Relative path to empty dir
            'val': 'val2017.txt',     # Relative path to our degraded images
            'test': '',                  # Optional
            'nc': num_classes,           # Number of classes
            'names': {i: name for i, name in enumerate(coco_class_names)}
        }
        try:
            with open(yaml_path, 'w') as f_yaml:
                yaml.dump(yaml_data, f_yaml, default_flow_style=False, sort_keys=False)
        except Exception as e:
             print(f"Error writing {yaml_path}: {e}")
             return None

        print(f"Degraded validation set '{degradation_name}' created successfully.")
        return str(yaml_path) # Return path to the yaml file

    except Exception as e:
        print(f"Error creating degraded dataset structure for '{degradation_name}': {e}")
        return None