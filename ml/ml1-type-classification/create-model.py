import os
import pandas as pd
from tqdm import tqdm
from PIL import Image
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import LabelEncoder

# -----------------------------
# 1. Setup dataset paths
# -----------------------------
dataset_path = "/Volumes/ryans-ssd/Datasets/clothing-dataset/images_resized"
labels_path = "/Volumes/ryans-ssd/Datasets/clothing-dataset/images.csv"

# -----------------------------
# 2. Load label CSV into DataFrame
# -----------------------------
df = pd.read_csv(labels_path)

# -----------------------------
# 3. Get list of image filepaths
# -----------------------------
image_filepaths = [
    os.path.join(dataset_path, filename)
    for filename in os.listdir(dataset_path)
    if filename.lower().endswith(('.jpg', '.jpeg')) and not filename.startswith('.')
]

# -----------------------------
# 4. Extract image IDs from filenames to match with labels
# -----------------------------
image_ids = [os.path.splitext(os.path.basename(fp))[0] for fp in image_filepaths]

# -----------------------------
# 5. Create a fast lookup dict for image ID → label
# -----------------------------
id_to_label = dict(zip(df['image'], df['label']))

# -----------------------------
# 6. Load images as normalized float32 tensors
# -----------------------------
def load_and_convert_to_tensor(image_path):
    # Open image and convert to RGB (3 channels)
    img = Image.open(image_path).convert('RGB')
    # Convert image to numpy array
    data = np.array(img, dtype=np.float32)
    # Normalize pixel values from [0, 255] to [0.0, 1.0]
    data /= 255.0
    # Convert numpy array to TensorFlow tensor
    img_tensor = tf.convert_to_tensor(data, dtype=tf.float32)
    return img_tensor

# Use tqdm for progress bar while loading images
X = [load_and_convert_to_tensor(fp) for fp in tqdm(image_filepaths, desc="Loading images")]

# -----------------------------
# 7. Create label list y aligned with X
# -----------------------------
y_str = [id_to_label.get(img_id, 'Unknown') for img_id in image_ids]

# -----------------------------
# 8. Encode string labels as numeric classes and keep mapping dictionary
# -----------------------------
le = LabelEncoder()
y = le.fit_transform(y_str)

# Mapping from numeric label -> class name
label_map = dict(zip(range(len(le.classes_)), le.classes_))

# -----------------------------
# 9. Print label mapping for verification
# -----------------------------
print("Label mapping (numeric_label -> class_name):")
for num_label, class_name in label_map.items():
    print(f"{num_label} -> {class_name}")

# -----------------------------
# 10. Final check: lengths match
# -----------------------------
assert len(X) == len(y), f"Number of images ({len(X)}) and labels ({len(y)}) must match!"
print(f"Loaded {len(X)} images and labels successfully.")

# -----------------------------
# From here, you can proceed to:
# - Build your TensorFlow Dataset or DataLoader
# - Define your model architecture (e.g. ResNet50 with 20-class output)
# - Train, evaluate, and predict as planned
# -----------------------------
