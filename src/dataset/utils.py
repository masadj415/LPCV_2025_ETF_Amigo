import torch
import torchvision
import pydantic
import os
import sys
import dataset.DatasetReader as DatasetReader
from pydantic import BaseModel, Field
from typing import List, Optional
from .transforms import TRANSFORMS 

# Names of the classes, given in order for TRACK 1
GLOBAL_CLASSES = [
    "Bicycle",
    "Car",
    "Motorcycle",
    "Airplane",
    "Bus",
    "Train",
    "Truck",
    "Boat",
    "Traffic Light",
    "Stop Sign",
    "Parking Meter",
    "Bench",
    "Bird",
    "Cat",
    "Dog",
    "Horse",
    "Sheep",
    "Cow",
    "Elephant",
    "Bear",
    "Zebra",
    "Backpack",
    "Umbrella",
    "Handbag",
    "Tie",
    "Skis",
    "Sports Ball",
    "Kite",
    "Tennis Racket",
    "Bottle",
    "Wine Glass",
    "Cup",
    "Knife",
    "Spoon",
    "Bowl",
    "Banana",
    "Apple",
    "Orange",
    "Broccoli",
    "Hot Dog",
    "Pizza",
    "Donut",
    "Chair",
    "Couch",
    "Potted Plant",
    "Bed",
    "Dining Table",
    "Toilet",
    "TV",
    "Laptop",
    "Mouse",
    "Remote",
    "Keyboard",
    "Cell Phone",
    "Microwave",
    "Oven",
    "Toaster",
    "Sink",
    "Refrigerator",
    "Book",
    "Clock",
    "Vase",
    "Teddy Bear",
    "Hair Drier",
]

CLASSES_COCO   = [s.lower() for s in GLOBAL_CLASSES] 
CLASSES_IMAGENET = [s.lower().replace(' ', '_') for s in GLOBAL_CLASSES] 

class DataloaderConfig(pydantic.BaseModel):
    batch_size: int = 1
    shuffle: bool = False
    num_workers: int = 0
    persistent_workers: bool = False
    pin_memory: bool = False
    prefetch_factor: int = 2
    drop_last: bool = False

class ModelConfig(pydantic.BaseModel):
    model_name: str
    version: str

class COCOConfig(BaseModel):
    path: Optional[str] = None  
    target_classes: List[str] = Field(default_factory=lambda: CLASSES_COCO) 
    transform : str = "PIL_TO_TENSOR"
    min_size: int = 70
    max_expand: float = 1.0

class ImagenetConfig(BaseModel):
    path: Optional[str] = None  
    target_classes: List[str] = Field(default_factory=lambda: CLASSES_IMAGENET) 
    transform: str = "PIL_TO_TENSOR"

class SDConfig(BaseModel):
    path: Optional[str] = None
    target_classes: List[str] = Field(default_factory=lambda: CLASSES_IMAGENET)
    transform: str = "PIL_TO_TENSOR"


class TrainingConfig(pydantic.BaseModel):
    """
    Configuration for training setup.
    """

    model_configuration         : ModelConfig

    coco_config_train           : COCOConfig
    coco_config_val             : COCOConfig
    imagenet_config_train       : ImagenetConfig
    imagenet_config_val         : ImagenetConfig
    SD_config_train             : SDConfig
    SD_config_val               : SDConfig
    styles_config_train         : SDConfig
    styles_config_val           : SDConfig

    dataloader_config           : DataloaderConfig

    label_smoothing             : float = 0

    balance_datasets_per_class : bool
    balance_dataset             : bool

    use_per_class_dataset_weighthing : bool = False

    dataset_weights             : List
    weight_decay                : float
    learning_rate               : float
    max_epochs                  : int
    devices                     : int


def get_dataloader(dataset, config: DataloaderConfig = None, shuffle=None):
    """
    Given the dataset and a DataloaderConfig, return a DataLoader.  

    Parameters
    ----------
    dataset : torch.utils.data.Dataset
    config : DataloaderConfig or None
    shuffle : bool or None

    Returns
    -------
        dataloader : torch.utils.data.DataLoader
    """
    if config is None:
        config = DataloaderConfig()

    return torch.utils.data.DataLoader(
        dataset,
        batch_size       = config.batch_size,
        shuffle          = shuffle if shuffle is not None else config.shuffle,
        num_workers      = config.num_workers,
        persistent_workers = config.persistent_workers if config.num_workers > 0 else False,
        pin_memory       = config.pin_memory,
        prefetch_factor  = None if config.num_workers == 0 else config.prefetch_factor,
        drop_last        = config.drop_last,
    )


def get_dataset_path(dataset_dir: str = None) -> str:
    """
    Returns the absolute path to the dataset directory.
    
    Args:
        dataset_dir (str, optional): User-specified dataset directory. Defaults to None.
    
    Returns:
        str: Absolute path to the dataset directory.
    """
    if dataset_dir is None:
        dataset_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../dataset"))
    
    if not os.path.exists(dataset_dir):
        raise FileNotFoundError(f"Dataset directory not found: {dataset_dir}")
    
    return dataset_dir


def get_imagenet_path(dataset_dir=None, mode="train"):
    """
    Returns the absolute path to the ImageNet dataset directory.

    Parameters
    ----------
    dataset_dir : str or None, optional
        Path to the root ImageNet dataset folder. If None, the default path is used:
        two directories above the current file, inside 'datasets/imagenet/<mode>'.

    mode : {'train', 'val'}, optional
        Specifies whether to return the path to the training or validation set.
        Defaults to 'train'.

    Returns
    -------
    str
        Absolute path to the specified ImageNet dataset folder.

    Raises
    ------
    FileNotFoundError
        If the constructed dataset path does not exist.
    """
    if dataset_dir is None:
        dataset_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), f"../../datasets/imagenet/{mode}"))

    if not os.path.exists(dataset_dir):
        raise FileNotFoundError(f"Dataset directory not found: {dataset_dir}")

    return dataset_dir


def get_imagenet_dataset(config = None, mode = "train"):
    """
    Returns a ImageNet dataset using the specified configuration and mode.
    Applies the selected transform and filters the dataset by target classes.
    """
    if(config == None):
        config = ImagenetConfig()
    img_path = get_imagenet_path(config.path, mode=mode)

    transform = TRANSFORMS[config.transform]

    dataset_reader = DatasetReader.CustomImageFolder(
        root_dir = img_path, 
        class_names = config.target_classes, 
        transform = transform,
        dataset_id=0
    )

    return dataset_reader

def get_stable_diffusion_path(dataset_dir = None, mode = "train"):
    """
    Returns the absolute path to the StableDiffusion dataset directory.

    Parameters
    ----------
    dataset_dir : str or None, optional
        Path to the root ImageNet dataset folder. If None, the default path is used:
        two directories above the current file, inside 'datasets/stable_diffusion/<mode>'.

    mode : {'train', 'val'}, optional
        Specifies whether to return the path to the training or validation set.
        Defaults to 'train'.

    Returns
    -------
    str
        Absolute path to the specified StableDiffusion dataset folder.

    Raises
    ------
    FileNotFoundError
        If the constructed dataset path does not exist.
    """

    if dataset_dir is None:
        dataset_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../datasets/stable_diffusion/" + mode))

    if not os.path.exists(dataset_dir):
        raise FileNotFoundError(f"Dataset directory not found: {dataset_dir}")
    
    return dataset_dir


def get_stable_diffusion_dataset(config = None, mode = "train"):
    """
    Returns a Stable Diffusion dataset using the specified configuration and mode.
    Applies the selected transform and filters the dataset by target classes.
    """
    if(config == None):
        config = SDConfig()
    img_path = get_stable_diffusion_path(config.path, mode=mode)

    transform = TRANSFORMS[config.transform]

    dataset_reader = DatasetReader.CustomImageFolder(
        root_dir = img_path, 
        class_names = config.target_classes, 
        transform = transform,
        dataset_id=2
    )

    return dataset_reader

def get_styles_path(dataset_dir = None, mode = "train"):
    """
    Returns the absolute path to the Styles dataset directory.

    Parameters
    ----------
    dataset_dir : str or None, optional
        Path to the root ImageNet dataset folder. If None, the default path is used:
        two directories above the current file, inside 'datasets/styles/<mode>'.

    mode : {'train', 'val'}, optional
        Specifies whether to return the path to the training or validation set.
        Defaults to 'train'.

    Returns
    -------
    str
        Absolute path to the specified Styles dataset folder.

    Raises
    ------
    FileNotFoundError
        If the constructed dataset path does not exist.
    """

    if dataset_dir is None:
        dataset_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../datasets/styles/" + mode))

    if not os.path.exists(dataset_dir):
        raise FileNotFoundError(f"Dataset directory not found: {dataset_dir}")
    
    return dataset_dir


def get_styles_dataset(config = None, mode = "train"):
    """
    Returns a Syles dataset using the specified configuration and mode.
    Applies the selected transform and filters the dataset by target classes.
    """

    if(config == None):
        config = SDConfig()
    img_path = get_styles_path(config.path, mode=mode)

    transform = TRANSFORMS[config.transform]

    dataset_reader = DatasetReader.CustomImageFolder(
        root_dir = img_path, 
        class_names = config.target_classes, 
        transform = transform,
        dataset_id=3
    )

    return dataset_reader

def get_new_coco_path(dataset_dir = None, mode = "train"):
    """
    Returns the absolute path to the COCO dataset directory.

    Parameters
    ----------
    dataset_dir : str or None, optional
        Path to the root ImageNet dataset folder. If None, the default path is used:
        two directories above the current file, inside 'datasets/coco_modified/<mode>'.

    mode : {'train', 'val'}, optional
        Specifies whether to return the path to the training or validation set.
        Defaults to 'train'.

    Returns
    -------
    str
        Absolute path to the specified COCO dataset folder.

    Raises
    ------
    FileNotFoundError
        If the constructed dataset path does not exist.
    """

    if dataset_dir is None:
        dataset_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../datasets/coco_modified/" + mode))

    if not os.path.exists(dataset_dir):
        raise FileNotFoundError(f"Dataset directory not found: {dataset_dir}")
    
    return dataset_dir


def get_new_coco_dataset(config = None, mode = "train"):
    """
    Returns a COCO dataset using the specified configuration and mode.
    Applies the selected transform and filters the dataset by target classes.
    """

    if(config == None):
        config = SDConfig()
    img_path = get_new_coco_path(config.path, mode=mode)

    transform = TRANSFORMS[config.transform]

    dataset_reader = DatasetReader.CustomImageFolder(
        root_dir = img_path, 
        class_names = config.target_classes, 
        transform = transform,
        dataset_id=1
    )

    return dataset_reader