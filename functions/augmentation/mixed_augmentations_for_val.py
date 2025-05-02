import os
import shutil
import yaml # Requires PyYAML: pip install pyyaml
import cv2 # pip install opencv-python
import numpy as np
import albumentations as A
import random
from pathlib import Path
from tqdm.auto import tqdm

# --- Augmentation Definitions (Use the same pool as for mixed calibration) ---
noise_var_low = (10/255, 30/255)
noise_var_med = (35/255, 55/255)
blur_kernel_low = (3, 5)
blur_kernel_med = (7, 11)
contrast_limit_low = (-0.6, -0.3) # Reduce contrast
jpeg_quality_mod = (50, 75)
jpeg_quality_heavy = (20, 45)
transform_noise_low = A.GaussNoise(std_range=noise_var_low, p=1.0, per_channel=False)
transform_noise_med = A.GaussNoise(std_range=noise_var_med, p=1.0, per_channel=False)
transform_blur_low = A.GaussianBlur(blur_limit=blur_kernel_low, p=1.0)
transform_blur_med = A.GaussianBlur(blur_limit=blur_kernel_med, p=1.0)
transform_contrast_low = A.RandomBrightnessContrast(brightness_limit=0, contrast_limit=contrast_limit_low, p=1.0)
transform_jpeg_mod = A.ImageCompression(quality_lower=jpeg_quality_mod[0], quality_upper=jpeg_quality_mod[1], compression_type="jpeg", p=1.0)
transform_jpeg_heavy = A.ImageCompression(quality_lower=jpeg_quality_heavy[0], quality_upper=jpeg_quality_heavy[1], compression_type="jpeg", p=1.0)

# Define the pool of degradations to randomly choose from
# Use A.OneOf to structure this nicely if applying ON the fly, but here we choose first
# For pre-generation, it's easier to have a list of transforms/types
degradation_transforms = {
    "noisy_low": transform_noise_low,
    "noisy_med": transform_noise_med,
    "blurry_low": transform_blur_low,
    "blurry_med": transform_blur_med,
    "contrast_low": transform_contrast_low,
    "jpeg_mod": transform_jpeg_mod,
    "jpeg_heavy": transform_jpeg_heavy,
}
degradation_names = list(degradation_transforms.keys())

# --- End Augmentation Definitions ---


def create_mixed_degradation_val_set(
    original_val_images_dir: str,
    original_coco_labels_dir: str, # Path to parent 'labels' dir
    output_dataset_root: str,
    degradation_ratio: float, # Ratio of images to apply *some* degradation to
    coco_class_names: list[str], # List of class names for data.yaml
    seed: int = 42,
):
    """
    Creates a COCO-like validation set containing a mix of clean images and
    images with one randomly selected degradation applied, saved with original filenames.

    Args:
        original_val_images_dir (str): Path to the original COCO val images (e.g., .../coco/images/val2017).
        original_coco_labels_dir (str): Path to the original COCO 'labels' directory.
        output_dataset_root (str): The root directory for the new mixed-degraded dataset structure.
        degradation_ratio (float): The target proportion of images within the set that will have
                                   a *random* degradation applied (0.0 to 1.0).
        coco_class_names (list[str]): List of COCO class names.
        seed (int): Random seed for reproducibility.

    Returns:
        str | None: Path to the generated data.yaml file if successful, None otherwise.
    """
    random.seed(seed)
    np.random.seed(seed) # For albumentations if it uses numpy random internally

    if not (0.0 <= degradation_ratio <= 1.0):
        print("Error: degradation_ratio must be between 0.0 and 1.0")
        return None

    orig_img_path = Path(original_val_images_dir)
    orig_lbl_path = Path(original_coco_labels_dir)
    root_path = Path(output_dataset_root)
    images_path = root_path / "images"
    labels_path = root_path / "labels"
    val_images_path = images_path / "val2017"
    val_labels_path = labels_path / "val2017"
    train_images_path = images_path / "train2017"

    dataset_name = f"coco_val_mixed_degrad_{int(degradation_ratio*100)}pct"
    print(f"\n--- Creating Mixed Degraded Validation Set: {dataset_name} ---")
    print(f"Output root: {root_path}")

    if not orig_img_path.is_dir():
        print(f"Error: Original validation image directory not found: {orig_img_path}")
        return None
    if not (orig_lbl_path / "val2017").is_dir():
         print(f"Error: Original validation labels directory ('val2017' subdir) not found within: {orig_lbl_path}")
         return None

    try:
        # Create directories
        root_path.mkdir(parents=True, exist_ok=True)
        images_path.mkdir(exist_ok=True)
        labels_path.mkdir(exist_ok=True)
        val_images_path.mkdir(exist_ok=True)
        val_labels_path.mkdir(exist_ok=True)
        train_images_path.mkdir(exist_ok=True)

        # --- Copy Labels (val2017 needed) ---
        print(f"Copying val2017 labels from {orig_lbl_path / 'val2017'}...")
        source_val_labels = orig_lbl_path / "val2017"
        for item in tqdm(os.listdir(source_val_labels), desc="Copying labels"):
             s = source_val_labels / item
             d = val_labels_path / item
             if s.is_file() and s.suffix == '.txt':
                 shutil.copy2(s, d)

        # --- Identify all original validation images ---
        original_image_files = sorted([p for p in orig_img_path.glob("*") if p.suffix.lower() in (".jpg", ".jpeg", ".png", ".bmp", ".webp")])
        num_total_images = len(original_image_files)
        if num_total_images == 0:
             print(f"Error: No images found in {orig_img_path}")
             return None

        # --- Decide which images to degrade ---
        num_to_degrade = int(round(num_total_images * degradation_ratio))
        indices = list(range(num_total_images))
        random.shuffle(indices)
        indices_to_degrade = set(indices[:num_to_degrade])
        print(f"Total validation images: {num_total_images}")
        print(f"Degradation ratio: {degradation_ratio:.2f}")
        print(f"Number of images to degrade: {num_to_degrade}")
        print(f"Number of clean images to copy: {num_total_images - num_to_degrade}")

        # --- Process all images: copy clean or apply random degradation ---
        processed_image_paths_relative = []
        processed_count = 0
        skipped_count = 0

        print(f"\nProcessing {num_total_images} validation images...")
        for i, img_file_path in enumerate(tqdm(original_image_files, desc="Processing val images")):
            destination_path = val_images_path / img_file_path.name

            if i in indices_to_degrade:
                # --- Degrade and Save ---
                try:
                    img = cv2.imread(str(img_file_path))
                    if img is None:
                        print(f"Warning: Could not read image {img_file_path}. Skipping.")
                        skipped_count += 1
                        continue

                    # --- Randomly select ONE degradation from the pool ---
                    chosen_degradation_name = random.choice(degradation_names)
                    chosen_transform = degradation_transforms[chosen_degradation_name]

                    augmented_data = chosen_transform(image=img)
                    degraded_img_np = augmented_data['image']

                    # Save degraded image with original name
                    success = cv2.imwrite(str(destination_path), degraded_img_np)
                    if not success:
                        print(f"Warning: Failed to save degraded image {destination_path}. Skipping.")
                        skipped_count += 1
                        continue # Skip adding to relative path list

                except Exception as e:
                    print(f"Error degrading or saving {img_file_path} as {destination_path}: {e}. Skipping.")
                    skipped_count += 1
                    continue # Skip adding to relative path list
            else:
                # --- Copy Clean ---
                try:
                    shutil.copy2(img_file_path, destination_path)
                except FileNotFoundError:
                     print(f"Warning: Source file not found, cannot copy: {img_file_path}. Skipping.")
                     skipped_count += 1
                     continue # Skip adding to relative path list
                except Exception as e:
                     print(f"Warning: Error copying {img_file_path} to {destination_path}: {e}. Skipping.")
                     skipped_count += 1
                     continue # Skip adding to relative path list

            # If processed/copied successfully, add relative path
            relative_path = destination_path.relative_to(root_path)
            processed_image_paths_relative.append(f"./{relative_path}")
            processed_count += 1

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
        yaml_path = root_path / f"data_{dataset_name}.yaml" # Unique yaml name
        print(f"Creating {yaml_path}...")
        num_classes = len(coco_class_names)
        yaml_data = {
            'path': str(root_path.resolve()), # Absolute path
            'train': 'images/train2017', # Relative path to empty dir
            'val': 'images/val2017',     # Relative path to our mixed images
            'test': '',
            'nc': num_classes,
            'names': {i: name for i, name in enumerate(coco_class_names)}
        }
        try:
            with open(yaml_path, 'w') as f_yaml:
                yaml.dump(yaml_data, f_yaml, default_flow_style=False, sort_keys=False)
        except Exception as e:
             print(f"Error writing {yaml_path}: {e}")
             return None

        print(f"Mixed degraded validation set '{dataset_name}' created successfully.")
        return str(yaml_path) # Return path to the yaml file

    except Exception as e:
        print(f"Error creating mixed degraded dataset structure for '{dataset_name}': {e}")
        return None