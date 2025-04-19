import os
import shutil
import random

train_dir = "/home/centar15-desktop1/LPCV_2025_T1/datasets/imagenet/train"
val_dir = "/home/centar15-desktop1/LPCV_2025_T1/datasets/imagenet/val"

# Iteriramo kroz sve klase u validation folderu
for class_name in os.listdir(val_dir):
    val_class_path = os.path.join(val_dir, class_name)
    train_class_path = os.path.join(train_dir, class_name)

    # Proveravamo da li je direktorijum
    if not os.path.isdir(val_class_path):
        continue

    # Broj slika u validation klasi
    val_images = os.listdir(val_class_path)
    if len(val_images) < 300:  # Ako ima manje od 300 slika u validaciji
        if not os.path.exists(train_class_path):
            print(f"Skipping {class_name}: No images in train folder.")
            continue

        train_images = os.listdir(train_class_path)
        if not train_images:
            print(f"Skipping {class_name}: No images left in train folder.")
            continue

        needed_images = min(300 - len(val_images), len(train_images))  # Koliko slika treba prebaciti
        images_to_move = random.sample(train_images, needed_images)  # Nasumično biranje slika

        # Premeštanje slika
        for img in images_to_move:
            src_path = os.path.join(train_class_path, img)
            dst_path = os.path.join(val_class_path, img)

            # Ako slika već postoji u val, dodajemo sufiks
            if os.path.exists(dst_path):
                name, ext = os.path.splitext(img)
                dst_path = os.path.join(val_class_path, f"{name}_train{ext}")

            shutil.move(src_path, dst_path)

        print(f"Moved {needed_images} random images from {train_class_path} to {val_class_path}")
