import torch
import torchsummary
import os
import sys
import pytorch_lightning as pl
import torch.optim as optim
import torch.nn as nn

import rcvit

class LightningModelCASVIT_T(pl.LightningModule):
    def __init__(self, num_classes=64, learning_rate=2e-5):

        super(LightningModelCASVIT_T, self).__init__()


        self.model = rcvit.rcvit_t()

        checkpoint = torch.load('CASVIT_t.pth') # Izmeniti
        self.model.load_state_dict(checkpoint["model"])

        self.model.dist = False
        self.model.head = nn.Linear(512, 64) # Menja classifier head da bude 64

        self.criterion = nn.CrossEntropyLoss()
        self.learning_rate = learning_rate
        self.alpha = 0.7

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        images, labels, ds = batch # ds = 0 - imagenet, ds = 1 - COCO
        images, labels = images.to(self.device), labels.to(self.device)
        outputs = self(images)
        loss = self.criterion(outputs, labels)

        ds = ds.to(self.device)
        alpha_mask = (ds == 0).float() * self.alpha + (ds == 1).float() * (1 - self.alpha)
        loss = loss * alpha_mask
        loss = loss.mean()

        self.log("train_loss", loss, prog_bar=True)
        return loss

    def configure_optimizers(self):
        return optim.Adam(self.parameters(), lr=self.learning_rate)