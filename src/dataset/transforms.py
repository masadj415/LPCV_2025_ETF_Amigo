from torchvision import transforms
from torchvision.transforms.autoaugment import RandAugment
import random

identity = transforms.Lambda(lambda x: x)  # No transformation
shear_x  = transforms.RandomAffine(degrees=0, shear=(-10, 10))        # ShearX in range [-10°, 10°]
shear_y  = transforms.RandomAffine(degrees=0, shear=(0, 0, -5, 5))    # Small ShearY in range [-5°, 5°]

AUGMENTATION = transforms.Compose([
    transforms.Lambda(lambda img: img.convert('RGB') if img.mode != 'RGB' else img),
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(p=0.3),
    transforms.ColorJitter(brightness=0.5),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

class ConvertToRGB:
    def __call__(self, img):
        return img.convert("RGB")

AUGMENTATION2 = transforms.Compose([
    ConvertToRGB(),
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(p=0.3),
    transforms.RandomEqualize(p=0.2),
    transforms.RandomRotation(degrees=30),
    transforms.ColorJitter(brightness=0.5, contrast=0.3, saturation=0.3, hue=0.03),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

AUGMENTATION3 = transforms.Compose([
    transforms.Lambda(lambda img: img.convert('RGB') if img.mode != 'RGB' else img),
    transforms.RandomChoice([identity, shear_x, shear_y]),
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomEqualize(p=0.2),
    transforms.RandomRotation(degrees=30),
    transforms.ColorJitter(brightness=0.5, contrast=0.3, saturation=0.3, hue=0.03),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

AUGMENTATION4 = transforms.Compose([
    transforms.Lambda(lambda img: img.convert('RGB') if img.mode != 'RGB' else img),
    transforms.RandomResizedCrop(size=(224, 224), scale=(0.5, 1.0), ratio=(0.5, 2.0)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomEqualize(p=0.2),
    transforms.RandomRotation(degrees=30),
    transforms.ColorJitter(brightness=0.5, contrast=0.3, saturation=0.3, hue=0.03),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

RANDAUG = transforms.Compose([
    transforms.Lambda(lambda img: img.convert('RGB') if img.mode != 'RGB' else img),
    transforms.RandAugment(num_ops=4, magnitude=9),  # Randomly applies 2 transformations
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

HIST_EQ_AUGMENT = transforms.Compose([
    transforms.Lambda(lambda img: img.convert('RGB') if img.mode != 'RGB' else img),
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(degrees=30),
    transforms.ColorJitter(brightness=0.5, contrast=0.3, saturation=0.3, hue=0.03),
    transforms.RandomEqualize(p=1),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.288, 0.288, 0.288])
])

# Transform that only converts PIL image to tensor
PIL_TO_TENSOR = transforms.Compose([
    transforms.Lambda(lambda img: img.convert('RGB') if img.mode != 'RGB' else img),
    transforms.ToTensor(),
])

# Transform used for inference 
RESIZE_NORMALIZE = transforms.Compose([
    ConvertToRGB(),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

HIST_EQ_RESIZE_NORMALIZE = transforms.Compose([
    transforms.Lambda(lambda img: img.convert('RGB') if img.mode != 'RGB' else img),
    transforms.Resize((224, 224)),
    transforms.RandomEqualize(p=1),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.288, 0.288, 0.288])
])

PIL_TO_TENSOR_RESIZE = transforms.Compose([
    transforms.Lambda(lambda img: img.convert('RGB') if img.mode != 'RGB' else img),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

TRANSFORMS = {
    "AUGMENTATION"              : AUGMENTATION,
    "AUGMENTATION2"             : AUGMENTATION2,
    "AUGMENTATION3"             : AUGMENTATION3,
    "AUGMENTATION4"             : AUGMENTATION4,
    "RANDAUG"                   : RANDAUG,
    "PIL_TO_TENSOR"             : PIL_TO_TENSOR,
    "RESIZE_NORMALIZE"          : RESIZE_NORMALIZE,
    "HIST_EQ_AUGMENT"           : HIST_EQ_AUGMENT,
    "HIST_EQ_RESIZE_NORMALIZE"  : HIST_EQ_RESIZE_NORMALIZE,
    "PIL_TO_TENSOR_RESIZE"      : PIL_TO_TENSOR_RESIZE
}
