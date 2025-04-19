import torch
import torchvision.transforms as T
from torchvision import datasets
import json
import os
from PIL import Image
import random
# import utils as utils

class COCODataset(torch.utils.data.Dataset):
    def __init__(self, annotation_file, image_dir, target_classes, transform=None, condition=None, min_size=30,max_expand=0.2):
        """
        Klasa koja daje COCO dataset. Neophodno je proslediti putanja fajla sa anotacijama, direktorijum sa svim slikama,
        zeljene klase (klase u JSON fajlu coco dataseta su date malim slovima, sa blanko karakterom za razmak)

        Minimalna velicina je minimalni broj horizontalnih i vertikalnih piksela koje treba da ima bounding box da bi anotacija bila ucitana
        max_expand se odnosi na to da se bounding box produzi za odredjeni (random uniformni faktor trenutno sirine/visine)

        Args:
            annotation_file (str): Path to the COCO JSON annotation file.
            image_dir (str): Path to the directory containing images.
            target_classes (list): List of category names to include.
            transform (callable, optional): Optional transform to apply to cropped images.
            Condition (callable, optional): Optional condition determine if the image should be included.
        """
        
        self.image_dir = image_dir
        self.transform = transform
        self.max_expand=max_expand
        
        # Load COCO annotations
        with open(annotation_file, 'r') as f:
            self.coco_data = json.load(f)
        

        # Zeljene kategorije
        self.target_cat_names = set(target_classes)


        self.name_to_annotation_id = {cat['name']: cat['id'] for cat in self.coco_data['categories']}
        self.annotation_id_to_name = {cat['id']: cat['name'] for cat in self.coco_data['categories']}
        self.target_cat_ids_annotations = {self.name_to_annotation_id[name] for name in target_classes} 


        self.name_to_local_id = {name: idx for idx, name in enumerate(target_classes)}
        self.local_id_to_name = {idx: name for idx, name in enumerate(target_classes)}
        
        # Create a mapping from image ID to image file
        self.image_id_to_file = {img['id']: img['file_name'] for img in self.coco_data['images']}
        

        self.annotations = self._filter_annotations_by_size(min_size)
    

        # Check the total number of annotations
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
        
        # print(type(cropped_img))
        # print(type(local_id))
        return cropped_img, local_id, 1
    
    def _filter_annotations_by_size(self, min_size):
        """Filters annotations based on bounding box size and prints stats for debugging."""
        filtered_annotations = []
        for ann in self.coco_data['annotations']:
            # Get width and height of the bounding box
            width, height = ann['bbox'][2], ann['bbox'][3]
            
            # Check if the annotation meets the conditions
            if ann['category_id'] in self.target_cat_ids_annotations and width >= min_size and height >= min_size:
                filtered_annotations.append(ann)
            else:
                # Debugging: Print annotations that fail the filtering
                pass
                #print(f"Excluded: Category: {ann['category_id']} | BBox: {width}x{height}")
        
        print(f"Total annotations before filtering: {len(self.coco_data['annotations'])}")
        print(f"Total annotations after filtering: {len(filtered_annotations)}")
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

        aspect_ratio = w / h  # Width-to-height ratio

        # Adjust expansion based on aspect ratio
        if aspect_ratio > 1:  # Wider than tall
            vertical_factor = 1.2  # Increase vertical expansion
            horizontal_factor = 0.8  # Reduce horizontal expansion
        elif aspect_ratio < 1:  # Taller than wide
            vertical_factor = 0.8  # Reduce vertical expansion
            horizontal_factor = 1.2  # Increase horizontal expansion
        else:
            vertical_factor = horizontal_factor = 1  # Already square

        # Apply the modified expansion factors
        left_expand = random.uniform(0, max_expand) * w * horizontal_factor
        right_expand = random.uniform(0, max_expand) * w * horizontal_factor
        top_expand = random.uniform(0, max_expand) * h * vertical_factor
        bottom_expand = random.uniform(0, max_expand) * h * vertical_factor

        x_new = max(0, x - int(left_expand))  # Expand left
        y_new = max(0, y - int(top_expand))  # Expand top
        x_max = min(img_width, x + w + int(right_expand))  # Expand right
        y_max = min(img_height, y + h + int(bottom_expand))  # Expand bottom

        w_new = x_max - x_new
        h_new = y_max - y_new

        return x_new, y_new, w_new, h_new

class CustomImageFolder(torch.utils.data.Dataset):
    def __init__(self, root_dir, class_names, transform=None, dataset_id = 0):
        
        """

        Klasa koja se koristi za pravljenje dataseta koji ima foldersku strukturu
        -root
        --class1
        ---imgName.jpg
        ---imgName.jpg
        ---imgName.jpg
        --class2
        ---imgName.jpg
        ---imgName.jpg
        --class3
         
        ...

        Bitna stvar kod ove klase je sto remapira imena klasa u brojeve, po redu koji su oni definisali
        na sajtu za track. Treba poslati imena klasa kao argument class_names, sto je niz po cijim indeksima
        se mapira ime_klase->id_klase (kod nas broj od 0 do 63). Bitno je da class_names budu ista kao imena foldera
        (gore dati kao class1, class2, class3 ...). U slucaju desktopa i jeta, to su na primer apple, traffic_light, sa
        full malim slovima i underscorovima kao razmak.

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
        # Get an image and label from the ImageFolder dataset
        image, label = self.image_folder[idx]
        
        # if self.transform:
        #     image = self.transform(image)
        # print(type(image))
        # print(type(label))
        return image, label, self.ds
