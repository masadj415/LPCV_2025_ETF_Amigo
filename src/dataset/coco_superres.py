import os
import cv2
import torch
import numpy as np
from torchvision import transforms
from pycocotools.coco import COCO
from torch.utils.data import DataLoader, Dataset
from super_image import EdsrModel, ImageLoader  # Super-resolution model
from tqdm import tqdm  # Progress bar
import random  # For bounding box expansion

from dataset.utils import CLASSES_COCO, get_coco_dataset, COCOConfig

# Configuration
COCO_ANNOTATIONS = "/home/centar15-desktop1/LPCV_2025_T1/datasets/coco/annotations/instances_val2017.json"
COCO_IMAGES = "/home/centar15-desktop1/LPCV_2025_T1/datasets/coco/val2017"
OUTPUT_DIR = "/home/centar15-desktop1/LPCV_2025_T1/datasets/coco_modified/val"
MODEL_NAME = "fsrcnn_x2"  # Can change to "espcn_x2"
BATCH_SIZE = 16
NUM_WORKERS = 12
FILTER_CLASSES = CLASSES_COCO

os.makedirs(OUTPUT_DIR, exist_ok=True)

# Load pre-trained super-resolution model
model = EdsrModel.from_pretrained('eugenesiow/edsr-base', scale=4)

# Dataset configuration
config = COCOConfig()
config.min_size = 35
config.max_expand = 0.75

# Load COCO dataset
coco_dataset = get_coco_dataset(config=config, mode='val')

# Create DataLoader
dataloader = DataLoader(
    coco_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=NUM_WORKERS
)

# Summary info
total_images = len(coco_dataset)
print(f"Total images in dataset: {total_images}")
print(f"Total classes in dataset: {len(coco_dataset.local_id_to_name)}")

# Initialize image counter per class
img_cnt = [0] * 100

# Process images one by one
for img, label_id, _ in tqdm(coco_dataset, total=total_images, desc="Processing Images"):
    img = img.unsqueeze(0).cuda()  # Add batch dimension and move to GPU

    with torch.no_grad():
        sr_img = model(img)  # Super-resolution

    # Convert output to image format
    sr_img = sr_img.cpu().squeeze().numpy()
    sr_img = np.transpose(sr_img, (1, 2, 0)) * 255
    sr_img = np.clip(sr_img, 0, 255).astype(np.uint8)

    # Determine save path
    class_label = coco_dataset.local_id_to_name[label_id]
    class_dir = os.path.join(OUTPUT_DIR, class_label)
    os.makedirs(class_dir, exist_ok=True)

    save_path = os.path.join(class_dir, f"{label_id}_{img_cnt[label_id]}.jpg")
    img_cnt[label_id] += 1

    # Save image
    cv2.imwrite(save_path, cv2.cvtColor(sr_img, cv2.COLOR_RGB2BGR))

print("Processing complete!")
