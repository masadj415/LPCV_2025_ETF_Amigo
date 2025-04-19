import torch
import torchvision.transforms as T
from torchvision import datasets
import json
import os
from PIL import Image
import random

class COCODataset(torch.utils.data.Dataset):
    def __init__(self, annotation_file, image_dir, target_classes, transform=None, min_size=30,max_expand=0.2):
        """
        COCO dataset wrapper.
        Args:
            annotation_file (str): Path to the COCO JSON annotation file.
            image_dir (str): Directory that contains all images.
            target_classes (list): List of category names to keep
                                   (in COCO JSON they are lower‑case with spaces).
            transform (callable, optional): Transform applied to the cropped object.
            min_size (int): Minimum width/height (in pixels) of the bounding box
                            for the annotation to be loaded.
            max_expand (float): Each side of the bounding box can be randomly
                                extended by up to this fraction of its width/height.
        """

        self.image_dir = image_dir
        self.transform = transform
        self.max_expand=max_expand
        
        # Load COCO annotations
        with open(annotation_file, 'r') as f:
            self.coco_data = json.load(f)
        
        # Wanted categories
        self.target_cat_names = set(target_classes)
        self.name_to_annotation_id = {cat['name']: cat['id'] for cat in self.coco_data['categories']}
        self.annotation_id_to_name = {cat['id']: cat['name'] for cat in self.coco_data['categories']}
        self.target_cat_ids_annotations = {self.name_to_annotation_id[name] for name in target_classes} 

        self.name_to_local_id = {name: idx for idx, name in enumerate(target_classes)}
        self.local_id_to_name = {idx: name for idx, name in enumerate(target_classes)}
        
        # Create a mapping from image ID to image file
        self.image_id_to_file = {img['id']: img['file_name'] for img in self.coco_data['images']}
    
        # Keep only annotations that satisfy size & class filters
        self.annotations = self._filter_annotations_by_size(min_size)
    
        # Debug stats
        total_annotations_coco = len(self.coco_data['annotations'])
        total_annotations_filtered = len(self.annotations)
        print(f"Total COCO annotations: {total_annotations_coco}")
        print(f"Filtered annotations: {total_annotations_filtered}")

        # Extract unique class IDs from both
        coco_classes = set(ann['category_id'] for ann in self.coco_data['annotations'])
        filtered_classes = set(ann['category_id'] for ann in self.annotations)
    
    
    def __len__(self):
        return len(self.annotations)
    
    def __getitem__(self, idx):
        """Returns a cropped object image and its label."""
        ann = self.annotations[idx]
        image_id = ann['image_id']
        category_id_annotation = ann['category_id']
        category_name = self.annotation_id_to_name[category_id_annotation]
        local_id = self.name_to_local_id[category_name]
        bbox = ann['bbox']  # Format: [x, y, width, height]
        
        # Load image
        image_path = os.path.join(self.image_dir, self.image_id_to_file[image_id])
        image = Image.open(image_path).convert("RGB")
        
        # Crop object
        x, y, w, h = self._random_expand_bbox(bbox,image.width,image.height,self.max_expand)

        cropped_img = image.crop((x, y, x + w, y + h))

        # Apply transformations if provided
        if self.transform:
            cropped_img = self.transform(cropped_img)

        return cropped_img, local_id, 1
    
    def _filter_annotations_by_size(self, min_size):
        """Keep annotations whose bbox ≥ min_size in both dimensions."""
        filtered_annotations = []
        for ann in self.coco_data["annotations"]:
            width, height = ann["bbox"][2], ann["bbox"][3]

            if (
                ann["category_id"] in self.target_cat_ids_annotations
                and width >= min_size
                and height >= min_size
            ):
                filtered_annotations.append(ann)

        print(f"Total annotations before filtering: {len(self.coco_data['annotations'])}")
        print(f"Total annotations after filtering:  {len(filtered_annotations)}")
        return filtered_annotations

    def _random_expand_bbox(self, bbox, img_width, img_height,max_expand=0.2):
        """
        Randomly adjusts the bounding box by expanding or shrinking each side independently.
        Args:
            bbox (list): Original bounding box [x, y, width, height].
            img_width (int): Width of the image.
            img_height (int): Height of the image.
        Returns:
            tuple: Adjusted bounding box (x, y, width, height).
        """
        x, y, w, h = bbox
        aspect_ratio = w / h  
    
        #Wider boxes -> expand more vertically, taller -> expand more horizontally
        if aspect_ratio > 1:  # wider
            vertical_factor, horizontal_factor = 1.2, 0.8
        elif aspect_ratio < 1:  # taller
            vertical_factor, horizontal_factor = 0.8, 1.2
        else:
            vertical_factor = horizontal_factor = 1.0
        left_expand = random.uniform(0, max_expand) * w * horizontal_factor
        right_expand = random.uniform(0, max_expand) * w * horizontal_factor
        top_expand = random.uniform(0, max_expand) * h * vertical_factor
        bottom_expand = random.uniform(0, max_expand) * h * vertical_factor

        x_new = max(0, x - int(left_expand))
        y_new = max(0, y - int(top_expand))
        x_max = min(img_width, x + w + int(right_expand))
        y_max = min(img_height, y + h + int(bottom_expand))

        return x_new, y_new, x_max - x_new, y_max - y_new

class CustomImageFolder(torch.utils.data.Dataset):
    def __init__(self, root_dir, class_names, transform=None, dataset_id = 0):
        """
        Builds a dataset from a directory tree:

        root/
        ├── class1/
        │   ├── img1.jpg
        │   ├── img2.jpg
        ├── class2/
        │   ├── img3.jpg
        │   └── ...
        ...

        The folder names must exactly match the COCO category names used in the
        LPCV track (lower‑case, spaces → underscores).  `class_names` must list
        those names **in the competition‑defined order**, so the index of each
        name (0‑63) is already the correct LPCV/COCO label.  The class remapping
        logic inside the constructor converts ImageFolder’s alphabetical labels
        to these official IDs.
    
        Args:
            root_dir (string): Directory with all the images, where each class is in a separate subfolder.
            class_names (list): List of class names, where the index of each class corresponds to the label.
            transform (callable, optional): Optional transform to be applied on an image.
        """
        self.ds = dataset_id
        self.root_dir = root_dir
        self.class_names = class_names
        self.transform = transform
        # Create the ImageFolder dataset
        self.image_folder = datasets.ImageFolder(root=root_dir, transform=transform, loader=Image.open)
        
        # Map folder names to their indices (labels)
        self.class_to_idx = {class_name: idx for idx, class_name in enumerate(class_names)}
        
        self.name_to_local_id = {name: idx for idx, name in enumerate(class_names)}
        self.imagenet_id_to_local_id = {}
        for cl in self.image_folder.classes:
            id_imgnet = self.image_folder.class_to_idx[cl]

            id_local = self.name_to_local_id[cl]
            self.imagenet_id_to_local_id[id_imgnet] = id_local

        # Update the labels in the image_folder dataset based on the provided class_names
        self.image_folder.samples = [
            (path, self.imagenet_id_to_local_id[class_id])
            for path, class_id in self.image_folder.samples
        ]
        
    def __len__(self):
        return len(self.image_folder)
    
    def __getitem__(self, idx):
        image, label = self.image_folder[idx]
        return image, label, self.ds
