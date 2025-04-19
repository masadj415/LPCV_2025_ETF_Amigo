import os
import shutil

# Define paths
train_dir = "/home/centar15-desktop1/LPCV_2025_T1/datasets/imagenet/train"
val_dir = "/home/centar15-desktop1/LPCV_2025_T1/datasets/imagenet/val"

# Iterate through all classes in validation
for class_name in os.listdir(val_dir):
    val_class_path = os.path.join(val_dir, class_name)
    train_class_path = os.path.join(train_dir, class_name)

    # Ensure it's a directory
    if not os.path.isdir(val_class_path):
        continue

    # Count images in validation class
    images = os.listdir(val_class_path)
    num_images = len(images)

    if num_images > 600:  # Move images only if more than 600 exist
        # Ensure train class exists
        os.makedirs(train_class_path, exist_ok=True)

        # Calculate how many images to move (leave 600 in val)
        num_to_move = num_images - 600

        # Move images from val to train
        for img in images[:num_to_move]:  # Move only the required number of images
            src_path = os.path.join(val_class_path, img)
            base, ext = os.path.splitext(img)
            
            # Generate a unique filename for the destination
            counter = 1
            dst_path = os.path.join(train_class_path, img)
            while os.path.exists(dst_path):
                dst_path = os.path.join(train_class_path, f"{base}_{counter}{ext}")
                counter += 1

            shutil.move(src_path, dst_path)

        print(f"Moved {num_to_move} images from {val_class_path} to {train_class_path}")
