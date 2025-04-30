import json
from pathlib import Path

def random_sample_from_dir(train_data: str, sample_size: int, seed: int = 42) -> list:
    """
    Randomly samples a specified number of image paths from the training directory with reproducibility.

    Args:
        train_data (str): The path to the training directory containing images.
        sample_size (int): The number of image paths to sample.
        seed (int, optional): The seed for the random number generator to ensure reproducibility.

    Returns:
        list: A list containing the sampled image paths.
    """
    import random

    # Get a list of all image paths in the directory
    image_extensions = (".jpg", ".jpeg", ".png", ".bmp")
    image_paths = [str(p) for p in Path(train_data).rglob("*") if p.suffix.lower() in image_extensions]

    if sample_size > len(image_paths):
        raise ValueError("Sample size cannot be larger than the number of images in the training directory.")
    
    # Set the random seed for reproducibility
    random.seed(seed)
    
    # Randomly sample image paths
    return random.sample(image_paths, sample_size)

if __name__ == "__main__":
    sample_sizes = [
        1250,
        2500,
        3750,
        5000,
    ]
    
    for sample_size in sample_sizes:
        rand_sample = random_sample_from_dir(
            train_data="/home/omni/Programming/QRID/datasets/coco/images/val2017",
            sample_size=sample_size,
            seed=42
        )
        save_path = Path("/home/omni/Programming/QRID/QRID/imgs_calibrated_for_ptq/random_sample_val_{}.json".format(sample_size))
        with open(save_path, "w") as f:
            json.dump(rand_sample, f, indent=4)  # Save the list as JSON