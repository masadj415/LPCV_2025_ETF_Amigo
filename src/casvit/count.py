
import os

def count_items_in_subfolders(root_folder):
    for subfolder in sorted(os.listdir(root_folder)):  # Sort for consistent order
        subfolder_path = os.path.join(root_folder, subfolder)
        if os.path.isdir(subfolder_path):
            item_count = len(os.listdir(subfolder_path))
            print(f"{subfolder}: {item_count} items")

# Example usage
root_folder = "/home/centar15-desktop1/LPCV_2025_T1/datasets/imagenet/val"  # Change this to your actual root folder path
count_items_in_subfolders(root_folder)
