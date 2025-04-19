import os
import shutil

def get_unique_filename(dest_dir, filename):
    """Generate a unique filename by appending a number if the file already exists."""
    base, ext = os.path.splitext(filename)
    counter = 1
    new_filename = filename

    while os.path.exists(os.path.join(dest_dir, new_filename)):
        new_filename = f"{base}_{counter}{ext}"
        counter += 1

    return new_filename

def move_images(source_root, destination_root):
    """Move images from source_root to destination_root, renaming if needed."""
    for class_name in os.listdir(source_root):
        source_class_dir = os.path.join(source_root, class_name)
        destination_class_dir = os.path.join(destination_root, class_name)

        if not os.path.isdir(source_class_dir):
            continue  # Skip if it's not a directory

        os.makedirs(destination_class_dir, exist_ok=True)

        moved_count = 0

        for image_name in os.listdir(source_class_dir):
            source_image_path = os.path.join(source_class_dir, image_name)

            if os.path.isfile(source_image_path):  # Ensure it's a file
                unique_name = get_unique_filename(destination_class_dir, image_name)
                destination_image_path = os.path.join(destination_class_dir, unique_name)

                shutil.move(source_image_path, destination_image_path)
                moved_count += 1

        print(f"Class '{class_name}': Moved {moved_count} images")

# Example usage
source_root = "/home/centar15-desktop1/Desktop/overwritten"
destination_root = "/home/centar15-desktop1/LPCV_2025_T1/datasets/coco_modified/train"

move_images(source_root, destination_root)
