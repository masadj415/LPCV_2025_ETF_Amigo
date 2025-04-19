import torch
from torchvision import models

import torch.nn as nn

class MobileNetV3_64_Classes(nn.Module):
    def __init__(self, pretrained=True):
        super(MobileNetV3_64_Classes, self).__init__()
        self.model = models.mobilenet_v3_large(pretrained=pretrained)
        # Replace the final classification layer
        self.model.classifier[3] = nn.Linear(self.model.classifier[3].in_features, 64)

    def forward(self, x):
        return self.model(x)
    

if __name__ == "__main__":

    from training.lightning_train_function import lightning_train

    model = MobileNetV3_64_Classes()
    # lightning_train("/Users/slowepoke/projects/lpcv/LPCV_2025_T1/configs/default_training_config.json", model)
    lightning_train('configs/default_training_config.json', model)

    
