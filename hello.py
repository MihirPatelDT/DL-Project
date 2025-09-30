import os
import random

# Path to your folder containing 21,000 images
folder_path = "number_plates_image\clear_images"

# Number of images to keep
keep_count = 8000

# Get all image file names
all_files = [f for f in os.listdir(folder_path) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff'))]

# Randomly select 8000 files to keep
files_to_keep = set(random.sample(all_files, keep_count))

# Delete the rest
for f in all_files:
    if f not in files_to_keep:
        os.remove(os.path.join(folder_path, f))

print(f"Kept {keep_count} images and deleted {len(all_files) - keep_count} images.")
