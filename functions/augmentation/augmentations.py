import albumentations as A
import random

# Define ranges for different levels of degradation
# (Adjust these values based on experimentation/visual inspection)
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
transform_jpeg_mod = A.ImageCompression(compression_type="jpeg", quality_range=jpeg_quality_mod, p=1.0)
transform_jpeg_heavy = A.ImageCompression(compression_type="jpeg", quality_range=jpeg_quality_heavy, p=1.0)

# Combine using OneOf to apply only ONE degradation per image randomly
# You can adjust the probabilities (e.g., p=0.25 for each type if 4 types)
# Or list more specific levels if desired.
apply_one_degradation = A.OneOf([
    transform_noise_low,       # Apply low noise sometimes
    transform_noise_med,       # Apply medium noise sometimes
    transform_blur_low,
    transform_blur_med,
    transform_contrast_low,
    transform_jpeg_mod,
    transform_jpeg_heavy,
], p=1.0) # p=1.0 ensures one of these is always chosen for the degraded images

# Assume 'paths_to_keep_clean' has 500 clean paths
# Assume 'degraded_paths' has 500 paths to the saved degraded images

mixed_calibration_paths = paths_to_keep_clean + degraded_paths
random.shuffle(mixed_calibration_paths) # Shuffle the final list

# Save this list to a JSON file (e.g., 'calibration_files_mixed.json')
# This JSON file will be used to initialize the CalibrationDataReader
# for your novel quantization method.