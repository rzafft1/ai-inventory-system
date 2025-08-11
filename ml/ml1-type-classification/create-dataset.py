"""
Purpose: Preprocess the original dataset and create a new dataset that can be used for the ML task

This script processes the original clothing dataset, which contains 5,000 high-resolution 
JPEG images of varying dimensions, and generates a resized version where all images 
are standardized to 224x224 pixels.

Output:
A new directory named 'images_resized' (created alongside the original dataset directory) 
will contain all resized images in JPEG format. 
- Non-square images will have their aspect ratios altered to match the fixed dimensions.
- The resizing process uses the LANCZOS resampling filter for high-quality downscaling.

Note:
This script is intended for preprocessing before training image classification models.
"""

import os
import pandas as pd
from tqdm import tqdm
from PIL import Image

# Step 1: Define paths
dataset_path = "/Volumes/ryans-ssd/Datasets/clothing-dataset/images_original"
resized_dataset_path = "/Volumes/ryans-ssd/Datasets/clothing-dataset/images_resized"

# Step 2: Create output folder for resized dataset if it doesn't exist
os.makedirs(resized_dataset_path, exist_ok=True)

# Step 3: Choose fixed size for all images in the dataset
target_size = (224, 224) 

# Step 4: Get all image filenames in the folder
image_files = [
    filename for filename in os.listdir(dataset_path)
    if filename.lower().endswith(('.jpg', '.jpeg')) and not filename.startswith('.')
]

# Step 5: Loop through images and resize
# Note: Use tqdm to show a progress bar for the process
# Note: Use PIL to load each image, convert to a consist color space, resize the image, and save the image
# Note: This process will change the aspect ratio of non-square images
for filename in tqdm(image_files, desc="Resizing images"):
    src_path = os.path.join(dataset_path, filename)
    dst_path = os.path.join(resized_dataset_path, filename)
    try:
        with Image.open(src_path) as img:
            img = img.convert("RGB")  
            img_resized = img.resize(target_size, Image.Resampling.LANCZOS)
            img_resized.save(dst_path, "JPEG", quality=95)
    except Exception as e:
        print(f"Error processing {filename}: {e}")

print("All images resized and saved.")

