import os
import cv2
import torch
import numpy as np
from torchvision import transforms
from pycocotools.coco import COCO
from torch.utils.data import DataLoader, Dataset
from super_image import EdsrModel , ImageLoader  # Super-resolution model
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
FILTER_CLASSES = CLASSES_COCO  # List of classes to process

os.makedirs(OUTPUT_DIR, exist_ok=True)

# Load COCO dataset
# coco = COCO(COCO_ANNOTATIONS)
model = EdsrModel.from_pretrained('eugenesiow/edsr-base', scale=4)  # Load pre-trained model

# Get category IDs for filtering
# cat_ids = coco.getCatIds(catNms=FILTER_CLASSES)

# print(cat_ids)
# print(len(cat_ids))


# # Dataloader
config = COCOConfig()
config.min_size = 35
config.max_expand = 0.75
# config.transform = "PIL_TO_TENSOR_RESIZE"
coco_dataset = get_coco_dataset(config = config, mode = 'val')

dataloader = DataLoader(coco_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)

total_images = len(coco_dataset)
print(f"Total images in dataset: {total_images}")
print(f"Total classes in dataset: {len(coco_dataset.local_id_to_name)}")
processed_images = 0

# Super-resolution and saving
# for images, labels, _ in tqdm(dataloader, total=len(dataloader), desc="Processing Images"):
#     sr_images = []
#     images = images.cuda()  # Move images to GPU for processing
#     with torch.no_grad():
#         # Perform super-resolution on the entire batch
#         sr_batch = model(images)
#         sr_batch = torch.nn.functional.interpolate(sr_batch, size=(224, 224), mode='bicubic')
#         sr_images = sr_batch.cpu().numpy()

#     # Save super-resolved images in class-based folders
#     for i, (sr_image, label) in enumerate(zip(sr_images, labels)):
#         class_label = coco_dataset.local_id_to_name[label]
#         class_dir = os.path.join(OUTPUT_DIR, class_label)
#         os.makedirs(class_dir, exist_ok=True)
#         save_path = os.path.join(class_dir, f"{processed_images}_{i}.jpg")
#         sr_image = np.transpose(sr_image.squeeze(), (1, 2, 0)) * 255  # Convert to image format
#         cv2.imwrite(save_path, cv2.cvtColor(sr_image.astype(np.uint8), cv2.COLOR_RGB2BGR))

#     processed_images += len(images)

img_cnt = [0]*100

for img, label_id, _ in tqdm(coco_dataset, total=len(coco_dataset), desc="Processing Images"):
    # Process one image at a time
    
    # Resize the image to match super-resolution model input size (224x224)
    img = img.unsqueeze(0).cuda()  # Add batch dimension and move to GPU
    
    # Run super-resolution model (no need to batch)
    with torch.no_grad():
        sr_img = model(img)
        # sr_img = torch.nn.functional.interpolate(sr_img, size=(224, 224), mode='bicubic')

    # Convert to numpy for saving
    sr_img = sr_img.cpu().squeeze().numpy()
    sr_img = np.transpose(sr_img, (1, 2, 0)) * 255  # Convert to image format

    # Convert the image back to the format required by cv2 for saving
    sr_img = np.clip(sr_img, 0, 255).astype(np.uint8)
    
    # Save the super-resolved image in class-based folders
    class_label = coco_dataset.local_id_to_name[label_id]  # Get class name from label_id
    class_dir = os.path.join(OUTPUT_DIR, class_label)
    os.makedirs(class_dir, exist_ok=True)
    
    save_path = os.path.join(class_dir, f"{label_id}_{img_cnt[label_id]}.jpg")
    img_cnt[label_id] += 1
    cv2.imwrite(save_path, cv2.cvtColor(sr_img, cv2.COLOR_RGB2BGR))


print("Processing complete!")
