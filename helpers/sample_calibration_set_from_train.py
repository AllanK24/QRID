import json
import random
import numpy as np
from pathlib import Path
from tqdm.auto import tqdm # Use tqdm for progress bars
from pycocotools.coco import COCO
from collections import defaultdict

def sample_coco_calibration_data(
    annotation_file: str,
    image_dir: str,
    target_sample_size: int,
    output_file: str,
    seed: int = 42, # for reproducibility
):
    """
    Samples images from the COCO dataset for calibration using pure random
    sampling followed by a top-up to ensure all classes are represented,
    with improved trimming logic.

    Args:
        annotation_file (str): Path to the COCO annotation JSON file (e.g., instances_train2017.json).
        image_dir (str): Path to the directory containing the COCO images.
        target_sample_size (int): The desired number of images in the calibration set.
        output_file (str): Path to save the resulting list of selected image file paths (as JSON).
        seed (int): Random seed for reproducibility.

    Returns:
        list[str]: A list of the full paths to the sampled images. None if an error occurs.
    """
    random.seed(seed)
    np.random.seed(seed)

    # --- 1. Load Annotations ---
    try:
        print(f"Loading annotations from {annotation_file}...")
        coco = COCO(annotation_file)
    except Exception as e:
        print(f"Error loading COCO annotations: {e}")
        return None

    image_dir_path = Path(image_dir)
    if not image_dir_path.is_dir():
        print(f"Error: Image directory not found at {image_dir}")
        return None

    # --- 2. Build Mappings ---
    print("Building image-category and category-image mappings...")
    img_ids_with_annos = sorted(list(coco.getImgIds()))
    if not img_ids_with_annos:
        print("Error: No image IDs found in annotations.")
        return None

    img_to_cats = defaultdict(set)
    cat_to_imgs = defaultdict(set)
    ann_ids = coco.getAnnIds(imgIds=img_ids_with_annos)
    if not ann_ids:
        print("Warning: No annotations found for the images listed in the annotation file.")

    for ann in tqdm(coco.loadAnns(ann_ids), desc="Processing annotations"):
        img_id = ann['image_id']
        cat_id = ann['category_id']
        if img_id in img_ids_with_annos:
            img_to_cats[img_id].add(cat_id)
            cat_to_imgs[cat_id].add(img_id)

    all_category_ids = set(cat_to_imgs.keys())
    num_categories = len(all_category_ids)
    print(f"Found {num_categories} unique categories with associated images.")

    # --- 3. Filter to Valid Image IDs & Check Target Size ---
    valid_image_ids = sorted([img_id for img_id in img_ids_with_annos if img_id in img_to_cats])
    if not valid_image_ids:
        print("Error: No valid images with linked category annotations found.")
        return None
    num_valid_images = len(valid_image_ids)
    print(f"Found {num_valid_images} images with linked category annotations.")

    # Check if target size is reasonable
    if target_sample_size < num_categories:
        print(f"Error: target_sample_size ({target_sample_size}) is less than the number of "
              f"categories ({num_categories}). Increase target_sample_size to ensure coverage.")
        return None # Or raise ValueError as suggested

    initial_sample_size = min(target_sample_size, num_valid_images)
    if initial_sample_size < target_sample_size:
        print(f"Warning: Requested sample size {target_sample_size} is larger than "
              f"available valid images ({num_valid_images}). Sampling {initial_sample_size}.")
    elif initial_sample_size <= 0:
         print(f"Error: Target sample size ({target_sample_size}) or available images ({num_valid_images}) is zero or less.")
         return None

    # --- 4. Perform Initial Pure Random Sampling ---
    print(f"Performing initial random sampling for {initial_sample_size} images...")
    # Sample directly into a list
    initial_random_sample_list = random.sample(valid_image_ids, initial_sample_size)

    # --- 5. Top-up Loop for Missing Classes ---
    print("Checking for missing classes and topping up...")
    selected_img_ids_set = set(initial_random_sample_list)
    categories_in_sample = {cat_id for img_id in selected_img_ids_set for cat_id in img_to_cats.get(img_id, set())}
    missing_category_ids = all_category_ids - categories_in_sample

    topups_list = [] # Keep track of images added during top-up
    if missing_category_ids:
        print(f"Missing {len(missing_category_ids)} categories. Adding examples...")
        for cid in sorted(list(missing_category_ids)):
            candidates = list(cat_to_imgs[cid] - selected_img_ids_set)
            if candidates:
                chosen_candidate = random.choice(candidates)
                topups_list.append(chosen_candidate)
                selected_img_ids_set.add(chosen_candidate) # Add to set to avoid re-adding for other missing cats
            else:
                print(f"  Warning: Category {cid} was missing, but all its images were already in the initial sample?")
        print(f"Added {len(topups_list)} images during top-up.")
    else:
        print("All categories represented in initial sample.")

    # --- 6. Combine and Trim (Improved Logic) ---
    combined_list = initial_random_sample_list + topups_list
    final_sample_size = len(combined_list)
    print(f"Total images after top-up: {final_sample_size}")

    final_selected_ids = []
    if final_sample_size <= target_sample_size:
        # No trimming needed, or we couldn't even reach the target (e.g., lack of candidates)
        final_selected_ids = combined_list
        if final_sample_size < target_sample_size:
             print(f"Warning: Final sample size {final_sample_size} is less than target {target_sample_size}.")
    else:
        # Trim intelligently, keeping all top-ups
        print(f"Sample size ({final_sample_size}) exceeds target ({target_sample_size}). Trimming...")
        num_to_keep_from_initial = target_sample_size - len(topups_list)
        if num_to_keep_from_initial < 0:
             print("Warning: More top-up images were added than the target sample size allows. "
                   "Resulting sample will contain only top-up images plus some initial random ones, "
                   f"totaling {target_sample_size}.")
             # This case is unlikely if the check in step 3 passed, but handle defensively.
             # We keep all topups and fill the rest randomly from initial.
             needed_from_initial = target_sample_size - len(topups_list) # Should be 0 if target < topups
             if needed_from_initial <= 0:
                 # If target size is <= number of topups, just keep required number of topups
                 final_selected_ids = random.sample(topups_list, target_sample_size)
             else:
                  # This path should not be reached if step 3 check is active
                  print("Error in trimming logic - unexpected state.")
                  final_selected_ids = random.sample(combined_list, target_sample_size) # Fallback
        else:
             # Select the required number randomly from the initial sample
             kept_initial = random.sample(initial_random_sample_list, num_to_keep_from_initial)
             # Combine with all top-up images
             final_selected_ids = kept_initial + topups_list
             print(f"Trimmed by keeping {num_to_keep_from_initial} from initial sample and all {len(topups_list)} top-ups.")

    print(f"Final selected image count: {len(final_selected_ids)}")

    # --- 7. Map Selected Image IDs to File Paths ---
    print("Mapping selected image IDs to file paths...")
    selected_image_paths = []
    # Load only the necessary image info using the FINAL list
    img_infos_dict = {img['id']: img for img in coco.loadImgs(final_selected_ids)}

    not_found_count = 0
    for img_id in tqdm(final_selected_ids, desc="Generating file paths"): # Iterate the FINAL list
        img_info = img_infos_dict.get(img_id)
        if img_info and 'file_name' in img_info:
            filename = img_info['file_name']
            full_path = image_dir_path / filename
            if full_path.is_file():
                selected_image_paths.append(str(full_path))
            else:
                not_found_count += 1
        else:
            print(f"Warning: Could not find filename info for selected image ID {img_id}")
            not_found_count += 1

    if not_found_count > 0:
         print(f"Warning: Could not find {not_found_count} image files corresponding to selected IDs.")

    final_found_count = len(selected_image_paths)
    print(f"Successfully mapped {final_found_count} image IDs to existing files.")

    if final_found_count == 0:
        print("Error: No valid image file paths could be generated. Check image directory and annotation integrity.")
        return None

    # --- 8. Save the List of Paths ---
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        print(f"Saving {final_found_count} sampled image paths to {output_path}...")
        with open(output_path, 'w') as f:
            json.dump(selected_image_paths, f, indent=4)
        print("Successfully saved output file.")
        return selected_image_paths
    except Exception as e:
        print(f"Error saving output file {output_path}: {e}")
        return None


# --- Example Usage ---
if __name__ == "__main__":
    # --- Configuration ---
    # IMPORTANT: Update these paths before running!
    coco_annotation_file = "/home/omni/Programming/QRID/datasets/coco/annotations/instances_train2017.json"
    coco_image_dir = "/home/omni/Programming/QRID/datasets/coco/images/train2017"
    calibration_sample_size = 1000 # Target number of images
    output_json_file = "/home/omni/Programming/QRID/QRID/imgs_calibrated_for_ptq/calibration_files_sampled_final.json" # Where to save the list
    # --- Run Sampling ---
    sampled_paths = sample_coco_calibration_data(
        annotation_file=coco_annotation_file,
        image_dir=coco_image_dir,
        target_sample_size=calibration_sample_size,
        output_file=output_json_file,
        seed=42 # Use a fixed seed for reproducibility
    )

    if sampled_paths:
        print(f"\nSuccessfully generated list of {len(sampled_paths)} calibration image paths.")