import json
def read_json_for_calibration(json_path: str) -> list:
    """
    Reads a JSON file and returns the data as a list of dictionaries.
    
    Args:
        json_path (str): Path to the JSON file.
        
    Returns:
        list: List of dictionaries containing the data from the JSON file.
    """
    try:
        print(f"Reading JSON file: {json_path}")
        with open(json_path, 'r') as f:
            image_paths = json.load(f)
            print(f"Loaded {len(image_paths)} image paths from JSON.")
    except FileNotFoundError:
        print(f"File not found: {json_path}")
        return []
    
    return image_paths