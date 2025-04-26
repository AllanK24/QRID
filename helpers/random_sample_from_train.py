from pathlib import Path

def random_sample_from_train(train_data: str, sample_size: int, seed: int = 42) -> list:
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