from pathlib import Path
from functions.augmentation.mixed_augmentations_for_val import create_mixed_degradation_val_set

if __name__ == "__main__":
    # --- Configuration ---
    ORIGINAL_COCO_DIR = Path("/home/omni/Programming/QRID/datasets/coco")
    ORIGINAL_VAL_IMAGES = ORIGINAL_COCO_DIR / "images" / "val2017"
    ORIGINAL_COCO_LABELS = ORIGINAL_COCO_DIR / "labels"

    OUTPUT_VAL_BASE_DIR = Path("./validation_datasets") # Where to create the dataset

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

    # --- Create a mixed validation set with 50% degraded images ---
    mix_ratio = 0.50
    output_dir_mixed = OUTPUT_VAL_BASE_DIR / f"coco_val_mixed_degrad_{int(mix_ratio*100)}pct"

    mixed_yaml_path = create_mixed_degradation_val_set(
        original_val_images_dir=str(ORIGINAL_VAL_IMAGES),
        original_coco_labels_dir=str(ORIGINAL_COCO_LABELS),
        output_dataset_root=str(output_dir_mixed),
        degradation_ratio=mix_ratio,
        coco_class_names=coco_class_names,
        seed=42
    )

    if mixed_yaml_path:
        print(f"\nSuccessfully created mixed validation dataset.")
        print(f"YAML file: {mixed_yaml_path}")
    else:
        print("\nFailed to create mixed validation dataset.")