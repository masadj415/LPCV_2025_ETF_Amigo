import numpy as np
from PIL import Image
import requests
import torch
import skimage as ski

from abc import ABC, abstractmethod

class input_getter(ABC):
    """ input_getter

    Apstraktna klasa za dobijanje ulaznih slika, za sada samo koristim random i onu njihovu sliku, ideja je da se kada se napravi 
    dataset da se nekako izvlaci iz njega 


    """
    @abstractmethod
    def get_input_torch(self):
        pass
    def get_input_numpy(self):
        pass


class random_input_getter(input_getter):
    def __init__(self):
        self.img = torch.rand(1, 3, 224, 224)
    def get_input_torch(self):
        return torch.rand(1, 3, 224, 224)
    def get_input_numpy(self):
        return self.img.numpy()

class mug_image_getter(input_getter):
    def __init__(self):
        sample_image_url = (
        "https://qaihub-public-assets.s3.us-west-2.amazonaws.com/apidoc/input_image1.jpg"
        )
        response = requests.get(sample_image_url, stream=True)
        response.raw.decode_content = True
        image = Image.open(response.raw).resize((224, 224))
        input_array = np.expand_dims(
            np.transpose(np.array(image, dtype=np.float32) / 255.0, (2, 0, 1)), axis=0
        )
        self.img = input_array
        self.img = ski.transform.resize(self.img, (1, 3, 224, 224))

    def get_input_torch(self):
        return torch.tensor(self.img)
    
    def get_input_numpy(self):
        return self.img
        
# Skroz glupa klasa koja uzima sliku na datom pathu

class local_image_getter():
    def __init__(self, path):
        self.img = ski.io.imread(path)
        self.img = ski.img_as_float32(self.img)
        self.img = np.expand_dims(np.transpose(self.img, (2, 0, 1)), axis=0)
        self.img = ski.transform.resize(self.img, (1, 3, 224, 224))

    def get_input_numpy(self):
        return self.img
    
    def get_input_torch(self):
        return torch.tensor(self.img)
    