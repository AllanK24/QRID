from pathlib import Path
import albumentations as A
from functions.augmentation.augmentations import create_single_degraded_val_set

if __name__ == "__main__":
    # --- Define the specific transforms you want to test ---

    transform_val_noisy_low = A.Compose([
        A.GaussNoise(std_range=(10/255, 30/255), p=1.0, per_channel=False)
    ])
    
    transform_val_noisy_med = A.Compose([
        A.GaussNoise(std_range=(35/255, 55/255), per_channel=False, p=1.0),
    ])

    # Example: Low Blur
    transform_val_blurry_low = A.Compose([
        A.GaussianBlur(blur_limit=(3, 5), p=1.0)
    ])
    
    # Example: Medium Blur
    transform_val_blurry_med = A.Compose([
        A.GaussianBlur(blur_limit=(7, 11), p=1.0)
    ])

    # Example: Heavy JPEG
    transform_val_jpeg_heavy = A.Compose([
        A.ImageCompression(quality_lower=20, quality_upper=45, compression_type="jpeg", p=1.0)
    ])

    # Example: Low Contrast
    transform_val_contrast_low = A.Compose([
        A.RandomBrightnessContrast(brightness_limit=0, contrast_limit=(-0.6, -0.3), p=1.0)
    ])

    # --- Add definitions for all degradations you want to evaluate ---
    
    # --- Configuration for Validation Set Creation ---
    ORIGINAL_COCO_DIR = Path("/home/omni/Programming/QRID/datasets/coco")
    ORIGINAL_VAL_IMAGES = ORIGINAL_COCO_DIR / "images" / "val2017" # Correct path to val images
    ORIGINAL_COCO_LABELS = ORIGINAL_COCO_DIR / "labels" # Path to parent 'labels' dir

    OUTPUT_VAL_BASE_DIR = Path("/home/omni/Programming/QRID/QRID/validation_datasets") # Where to create the degraded sets

    # Load or define your COCO class names
    coco_class_names = [ # Example COCO names - replace if needed
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

    # --- List of degradations to create ---
    degradations_to_create = {
        "noisy_low": transform_val_noisy_low,
        "noisy_medium": transform_val_noisy_med,
        "blurry_low": transform_val_blurry_low,
        "blurry_medium": transform_val_blurry_med, # Add transform if defined
        "jpeg_heavy": transform_val_jpeg_heavy,
        "contrast_low": transform_val_contrast_low,
        # Add more entries here for each degradation type/level
    }

    generated_yaml_files = {}

    # --- Create each degraded set ---
    for name, transform in degradations_to_create.items():
        output_dir = OUTPUT_VAL_BASE_DIR / f"coco_val_{name}"
        yaml_path = create_single_degraded_val_set(
            original_val_images_dir=str(ORIGINAL_VAL_IMAGES),
            original_coco_labels_dir=str(ORIGINAL_COCO_LABELS),
            output_dataset_root=str(output_dir),
            degradation_name=name,
            albumentations_transform=transform,
            coco_class_names=coco_class_names
        )
        if yaml_path:
            generated_yaml_files[name] = yaml_path
            print(f"Successfully created dataset for '{name}'. YAML: {yaml_path}")
        else:
            print(f"!!! Failed to create dataset for '{name}' !!!")

    # --- Now you have the YAML files for benchmarking ---
    print("\n--- Degraded Validation Set Creation Complete ---")
    print("Generated YAML files for benchmarking:")
    for name, path in generated_yaml_files.items():
        print(f"  {name}: {path}")

    # You will pass these generated YAML paths to the 'data' argument
    # of your benchmark function when evaluating models on degraded data.
    # Don'g forget to also benchmark against the ORIGINAL clean COCO val yaml.