import os
import shutil

# Define paths
train_dir = "/home/centar15-desktop1/LPCV_2025_T1/datasets/coco_modified/train"
val_dir = "/home/centar15-desktop1/LPCV_2025_T1/datasets/coco_modified/val"

# Iterate through all classes in validation
for class_name in os.listdir(val_dir):
    val_class_path = os.path.join(val_dir, class_name)
    train_class_path = os.path.join(train_dir, class_name)

    # Ensure it's a directory
    if not os.path.isdir(val_class_path):
        continue

    # Count images in validation class
    images = os.listdir(val_class_path)
    if len(images) > 300:
        # Ensure train class exists
        os.makedirs(train_class_path, exist_ok=True)
        
        # Move images from val to train
        for img in images:
            src_path = os.path.join(val_class_path, img)
            dst_path = os.path.join(train_class_path, img)
            
            # Ensure unique name
            base, ext = os.path.splitext(img)
            counter = 1
            while os.path.exists(dst_path):
                dst_path = os.path.join(train_class_path, f"{base}_{counter}{ext}")
                counter += 1
            
            shutil.move(src_path, dst_path)
        
        print(f"Moved {len(images)} images from {val_class_path} to {train_class_path}")
