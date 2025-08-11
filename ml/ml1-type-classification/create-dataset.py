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

#  Define paths
dataset_path = "C:/Users/rzafft1/Desktop/images_original"
resized_dataset_path = "C:/Users/rzafft1/Desktop/images_resized"
os.makedirs(resized_dataset_path, exist_ok=True)

# Target size
target_size = (224, 224)

# Supported input image extensions
supported_exts = ('.jpg', '.jpeg', '.png', '.jfif', '.bmp', '.tiff', '.webp')

# Get all image filenames with supported extensions, ignoring hidden files
image_files = [
    f for f in os.listdir(dataset_path)
    if f.lower().endswith(supported_exts) and not f.startswith('.')
]

for filename in tqdm(image_files, desc="Resizing and converting images"):
    src_path = os.path.join(dataset_path, filename)
    
    # Change extension to .jpg for saving
    base_name = os.path.splitext(filename)[0]
    dst_filename = base_name + ".jpg"
    dst_path = os.path.join(resized_dataset_path, dst_filename)
    
    try:
        with Image.open(src_path) as img:
            img = img.convert("RGB")  # Convert to RGB color space for JPEG
            img_resized = img.resize(target_size, Image.Resampling.LANCZOS)
            img_resized.save(dst_path, "JPEG", quality=95)
    except Exception as e:
        print(f"Error processing {filename}: {e}")

print("All images resized and converted to JPEG.")
