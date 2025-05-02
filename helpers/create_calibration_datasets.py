import os
import shutil
import yaml # Requires PyYAML: pip install pyyaml
from pathlib import Path
from tqdm.auto import tqdm
# Assuming other imports like random_sample_from_dir and create_mixed_calibration_set are handled

def create_coco_calib_structure(
    output_dataset_root: str,
    image_paths_for_val: list[str], # List of SOURCE image paths
    original_coco_labels_dir: str,
    original_coco_train_txt: str,
    dataset_name: str = "CalibrationSet"
):
    """
    Creates a COCO-like dataset structure for calibration/validation purposes.
    Corrected val2017.txt generation.

    Args:
        output_dataset_root (str): The root directory for the new dataset structure.
        image_paths_for_val (list[str]): List of absolute paths to the SOURCE images
                                        that should be copied into images/val2017/.
        original_coco_labels_dir (str): Path to the original COCO 'labels' directory
                                         (containing train2017, val2017 subdirs, etc.).
        original_coco_train_txt (str): Path to the original COCO 'train2017.txt' file.
        dataset_name (str): A name for the dataset (used in data.yaml).

    Returns:
        bool: True if successful, False otherwise.
    """
    root_path = Path(output_dataset_root)
    images_path = root_path / "images"
    labels_path = root_path / "labels"
    val_images_path = images_path / "val2017"
    train_images_path = images_path / "train2017" # Empty dir needed

    print(f"\nCreating dataset structure at: {root_path}")

    try:
        # Create directories
        root_path.mkdir(parents=True, exist_ok=True)
        images_path.mkdir(exist_ok=True)
        labels_path.mkdir(exist_ok=True)
        val_images_path.mkdir(exist_ok=True)
        train_images_path.mkdir(exist_ok=True) # Create empty train images dir

        # --- Copy Labels ---
        print(f"Copying labels from {original_coco_labels_dir} to {labels_path}...")
        source_labels_path = Path(original_coco_labels_dir) # Use Path object
        if not source_labels_path.is_dir():
             print(f"Error: Original COCO labels directory not found at {source_labels_path}")
             return False
        # Copy the entire labels directory content
        shutil.copytree(source_labels_path, labels_path, dirs_exist_ok=True)

        # --- Copy train2017.txt ---
        print(f"Copying {original_coco_train_txt}...")
        train_txt_src_path = Path(original_coco_train_txt)
        if not train_txt_src_path.is_file():
            print(f"Error: Original train2017.txt not found at {train_txt_src_path}")
            return False
        shutil.copy2(train_txt_src_path, root_path / train_txt_src_path.name)

        # --- Copy Images for Validation Set and Collect Relative Paths ---
        print(f"Populating {val_images_path} with {len(image_paths_for_val)} images...")
        processed_count = 0
        successfully_copied_relative_paths = [] # Store relative paths of successfully copied files
        for src_img_path_str in tqdm(image_paths_for_val, desc="Copying val images"):
            src_img_path = Path(src_img_path_str)
            dest_img_path = val_images_path / src_img_path.name # Use original filename
            try:
                 if not src_img_path.is_file():
                      print(f"Warning: Source image not found {src_img_path}. Skipping.")
                      continue
                 # Copy the image (clean or degraded) into the val2017 folder
                 shutil.copy2(src_img_path, dest_img_path)
                 # If copy successful, add its relative path to the list
                 relative_path = dest_img_path.relative_to(root_path)
                 successfully_copied_relative_paths.append(f"./{relative_path}") # Correct format
                 processed_count += 1
            except Exception as e:
                 print(f"Warning: Failed to copy {src_img_path} to {dest_img_path}: {e}. Skipping.")

        if processed_count == 0 and len(image_paths_for_val) > 0:
             print("Error: Failed to copy any validation images.")
             return False
        print(f"Copied {processed_count} images to {val_images_path}.")

        # --- Create val2017.txt --- ## FIXED LOGIC ##
        val_txt_path = root_path / "val2017.txt"
        print(f"Creating {val_txt_path} with {len(successfully_copied_relative_paths)} entries...")
        try:
            # Sort the relative paths for consistent order
            successfully_copied_relative_paths.sort()
            with open(val_txt_path, 'w') as f_val:
                # Write each successfully copied relative path ONCE
                for rel_path in successfully_copied_relative_paths:
                    f_val.write(f"{rel_path}\n") # Write the collected relative path
        except Exception as e:
             print(f"Error writing {val_txt_path}: {e}")
             return False

        # --- Create data.yaml ---
        yaml_path = root_path / "data.yaml"
        print(f"Creating {yaml_path}...")
        # Assuming standard COCO 80 classes - replace if needed
        coco_class_names = [
            'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 'traffic light',
            'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
            'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee',
            'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard',
            'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
            'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch',
            'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote', 'keyboard',
            'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase',
            'scissors', 'teddy bear', 'hair drier', 'toothbrush'
        ]
        num_classes = len(coco_class_names)
        yaml_data = {
            'path': str(root_path.resolve()), # Absolute path
            # **Corrected paths for YAML**
            'train': 'images/train2017', # Point to image directory
            'val': 'images/val2017',     # Point to image directory
            'test': '',
            'nc': num_classes,
            'names': {i: name for i, name in enumerate(coco_class_names)}
        }
        try:
            with open(yaml_path, 'w') as f_yaml:
                yaml.dump(yaml_data, f_yaml, default_flow_style=False, sort_keys=False)
        except Exception as e:
             print(f"Error writing {yaml_path}: {e}")
             return False

        print(f"Dataset structure created successfully at {root_path}")
        return True

    except Exception as e:
        print(f"Error creating dataset structure at {root_path}: {e}")
        return False


# --- The orchestration script below remains the same ---
if __name__ == "__main__":
    # ... (rest of your main script calling create_coco_calib_structure) ...
    # Assume your previously defined functions are importable/available:
    from helpers.random_sample_from_dir import random_sample_from_dir
    from functions.augmentation.augmentations import create_mixed_calibration_set

    # --- Configuration ---
    ORIGINAL_COCO_DIR = Path("/home/omni/Programming/QRID/datasets/coco")
    ORIGINAL_COCO_TRAIN_IMAGES = ORIGINAL_COCO_DIR / "images" / "train2017"
    ORIGINAL_COCO_LABELS = ORIGINAL_COCO_DIR / "labels" # Path to parent 'labels' dir
    ORIGINAL_COCO_TRAIN_TXT = ORIGINAL_COCO_DIR / "train2017.txt"

    OUTPUT_BASE_DIR = Path("/home/omni/Programming/QRID/QRID/calibration_sets") # Where to create the new dataset folders
    TEMP_MIXED_IMAGES_DIR = Path("./temp_mixed_images_for_calib") # Intermediate storage

    SAMPLE_SIZE = 1000
    MIXED_SET_DEGRADATION_RATIO = 0.5 # 50% degraded
    RANDOM_SEED = 42

    # --- Step 1: Sample Clean Images ---
    print("--- Sampling Clean Calibration Images ---")
    clean_sampled_paths = random_sample_from_dir(
        train_data=str(ORIGINAL_COCO_TRAIN_IMAGES),
        sample_size=SAMPLE_SIZE,
        seed=RANDOM_SEED
    )

    if not clean_sampled_paths:
        print("Failed to sample clean images. Exiting.")
        exit()

    print(f"Sampled {len(clean_sampled_paths)} clean image paths.")

    # --- Step 2: Create Clean Calibration Dataset Structure ---
    print("\n--- Creating Clean Calibration Dataset Structure ---")
    clean_calib_root = OUTPUT_BASE_DIR / "coco_calib_clean"
    success_clean = create_coco_calib_structure(
        output_dataset_root=str(clean_calib_root),
        image_paths_for_val=clean_sampled_paths, # Use the clean sampled paths directly
        original_coco_labels_dir=str(ORIGINAL_COCO_LABELS),
        original_coco_train_txt=str(ORIGINAL_COCO_TRAIN_TXT),
        dataset_name="COCO_Calib_Clean"
    )

    if not success_clean:
        print("Failed to create clean calibration dataset structure. Exiting.")
        exit()

    print(f"Clean calibration dataset ready at: {clean_calib_root}")
    print(f"Use YAML: {clean_calib_root / 'data.yaml'}")


    # --- Step 3: Generate Mixed Images (Clean + Degraded) ---
    print("\n--- Generating Mixed Calibration Images (Degraded + Clean Copies) ---")
    mixed_image_paths_in_temp = create_mixed_calibration_set(
        clean_image_paths=clean_sampled_paths, # Use the same clean sample as input
        output_dir=str(TEMP_MIXED_IMAGES_DIR),
        target_total_size=SAMPLE_SIZE, # Target the same total size
        degradation_ratio=MIXED_SET_DEGRADATION_RATIO,
        seed=RANDOM_SEED
    )

    if not mixed_image_paths_in_temp:
        print("Failed to generate mixed calibration images. Exiting.")
        exit()

    print(f"Generated {len(mixed_image_paths_in_temp)} mixed images in {TEMP_MIXED_IMAGES_DIR}.")

    # --- Step 4: Create Mixed Calibration Dataset Structure ---
    print("\n--- Creating Mixed Calibration Dataset Structure ---")
    mixed_calib_root = OUTPUT_BASE_DIR / "coco_calib_mixed"
    success_mixed = create_coco_calib_structure(
        output_dataset_root=str(mixed_calib_root),
        image_paths_for_val=mixed_image_paths_in_temp, # Use paths generated by augmentation func
        original_coco_labels_dir=str(ORIGINAL_COCO_LABELS),
        original_coco_train_txt=str(ORIGINAL_COCO_TRAIN_TXT),
        dataset_name="COCO_Calib_Mixed"
    )

    if not success_mixed:
        print("Failed to create mixed calibration dataset structure. Exiting.")
        exit()

    print(f"Mixed calibration dataset ready at: {mixed_calib_root}")
    print(f"Use YAML: {mixed_calib_root / 'data.yaml'}")

    # --- Step 5: Optional Cleanup ---
    print(f"\nOptionally removing temporary mixed images directory: {TEMP_MIXED_IMAGES_DIR}")
    shutil.rmtree(TEMP_MIXED_IMAGES_DIR, ignore_errors=True) # Uncomment to enable cleanup

    print("\n--- Calibration Dataset Creation Complete ---")