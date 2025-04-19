import torch
import torchsummary
import os
import sys
import pytorch_lightning as pl
import torch.optim as optim
import torch.nn as nn

ROOT_DIR = os.path.abspath(os.path.join(os.getcwd(), ".."))
sys.path.append(ROOT_DIR)

ROOT_DIR = os.path.abspath(os.path.join(os.getcwd(), "../.."))
sys.path.append(ROOT_DIR)

from utils import helper, input_getter, qai_hub_jobs, tfhelper
import training
import casvit.rcvit as rcvit

def getBaseModel():
    modelBase = rcvit.rcvit_t()
    weights = torch.randn([64])
    modelBase.dist = False
    modelBase.head = nn.Linear(512, 64)  # Replace classifier head with 64-class output
    return modelBase

import training.lightning_model
import training.lightning_train_function

modelBase = getBaseModel()

lightningModel = training.lightning_model.LightningModel.load_from_checkpoint(
    r"../../finalLightningCheckpoint/FINAL_MODEL-epoch=54-train_loss=0.6094-val_loss=0.8717.ckpt",
    model=modelBase
)

model = lightningModel.model
model.eval()

from dataset import DatasetReader
from dataset.utils import GLOBAL_CLASSES
from torchvision import transforms
from torch.utils.data import DataLoader
import torchvision.ops as ops
import torchvision.transforms.functional as F

class Normalized(torch.nn.Module):
    def __init__(self, network):
        super(Normalized, self).__init__()
        self.model = network
        self.preprocess = transforms.Compose([
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])

    def forward(self, img):
        return self.model(self.preprocess(img))

modelOriginal = model
model = Normalized(modelOriginal)

import qai_hub

input_shape = (1, 3, 224, 224)
dummy_input = torch.randn(1, 3, 224, 224)
pt_model = torch.jit.trace(model.cpu().eval(), dummy_input)

compile_job = qai_hub.submit_compile_job(
    pt_model,
    name="Final Model epoch 54",
    device=qai_hub.Device("Snapdragon 8 Elite QRD"),
    input_specs=dict(image=input_shape)
)

compile_job.modify_sharing(add_emails=['lowpowervision@gmail.com'])  