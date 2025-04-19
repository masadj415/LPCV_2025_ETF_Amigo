import os
import shutil
from hashlib import md5

from PIL import Image
import numpy as np

def hash_image(image_path):
    """Compute MD5 hash of an image."""
    with open(image_path, "rb") as f:
        return md5(f.read()).hexdigest()

def are_images_identical(img1_path, img2_path):
    """Check if two images are identical."""
    return hash_image(img1_path) == hash_image(img2_path)

def process_images(root1, root2, output_root1, output_root2):
    """Compare images between root1 and root2, categorizing them based on similarity."""
    for class_name in os.listdir(root2):
        class_dir2 = os.path.join(root2, class_name)
        class_dir1 = os.path.join(root1, class_name)

        if not os.path.isdir(class_dir2):
            continue

        for image_name in os.listdir(class_dir2):
            img_path2 = os.path.join(class_dir2, image_name)
            img_path1 = os.path.join(class_dir1, image_name) if os.path.exists(class_dir1) else None

            if img_path1 and os.path.exists(img_path1):
                if are_images_identical(img_path1, img_path2):
                    dest_dir = os.path.join(output_root1, class_name)
                else:
                    dest_dir = os.path.join(output_root2, class_name)
            else:
                continue  # Ignore images that don't have a match in root1

            os.makedirs(dest_dir, exist_ok=True)
            shutil.copy(img_path2, os.path.join(dest_dir, image_name))

root1 = "/home/centar15-jet/LPCV_2025_T1/datasets/coco_modified/train"
root2 = "/home/centar15-jet/Desktop/zeznuli_sve"
output_root1 = "/home/centar15-jet/Desktop/comeback/saved"
output_root2 = "/home/centar15-jet/Desktop/comeback/overwritten"

process_images(root1, root2, output_root1, output_root2)