import torch
import torch.nn as nn
import pytorch_lightning as pl
from torch.optim.lr_scheduler import CosineAnnealingLR
from torchvision import transforms
from timm.data import RandAugment, Mixup

torch.set_printoptions(profile="full")


UNIFORM_WEIGHTS = torch.ones([64])
UNIFORM_TABLE   = torch.ones([64, 4])

class LightningModelScratch(pl.LightningModule):
    def __init__(self, model, learning_rate=6e-3, dataset_weights=[0.7, 0.3, 0.0, 0.0], class_weights=UNIFORM_WEIGHTS, weight_decay=0.01, table=UNIFORM_TABLE, label_smoothing=0.1):
        super().__init__()
        if class_weights is None:
            class_weights = UNIFORM_WEIGHTS
        if table is None:
            table = UNIFORM_TABLE
        
        self.model = model
        self.criterion = nn.CrossEntropyLoss(weight=torch.tensor(class_weights), reduction="none", label_smoothing=label_smoothing)
        self.learning_rate = learning_rate
        self.dataset_weights = torch.tensor(dataset_weights, requires_grad=False)
        self.table = torch.tensor(table, requires_grad=False)
        self.weight_decay = weight_decay

        # Mixup & CutMix setup
        self.mixup_fn = Mixup(
            mixup_alpha=0.8, cutmix_alpha=1.0, prob=1.0, switch_prob=0.5, mode="batch", label_smoothing=label_smoothing, num_classes=64
        )

    def on_fit_start(self):
        self.dataset_weights = self.dataset_weights.to(self.device)
        self.table = self.table.to(self.device)
        
        print("Training Initialized with the following settings:")
        print(f"Learning Rate: {self.learning_rate}")
        print(f"Weight Decay: {self.weight_decay}")
        print(f"Dataset Weights: {self.dataset_weights.tolist()}")
        print(f"Label Smoothing: {self.criterion.label_smoothing}")
        print("Mixup & CutMix Enabled")

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        images, labels, ds = batch  
        images, labels = images.to(self.device), labels.to(self.device)

        if self.mixup_fn is not None:
            images, labels = self.mixup_fn(images, labels)
        
        # print(labels.shape)
        # print(labels)

        outputs = self(images)
        loss = self.criterion(outputs, labels)
        ds = ds.to(self.device)
        loss = loss * self.dataset_weights[ds]
        loss = loss.mean()
    
        self.log("train_loss", loss, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        images, labels, ds = batch
        images, labels = images.to(self.device), labels.to(self.device)
        outputs = self(images)
        loss = self.criterion(outputs, labels)
        ds = ds.to(self.device)
        loss = loss * self.dataset_weights[ds] * self.table[labels, ds]
        loss = loss.mean()

        preds = torch.argmax(outputs, dim=1)
        acc = (preds == labels).float().mean()

        self.log("val_loss", loss, prog_bar=True, on_epoch=True)
        self.log("val_acc", acc, prog_bar=True, on_epoch=True)
        return {"val_loss": loss, "val_acc": acc}

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)
        scheduler = {
            "scheduler": CosineAnnealingLR(optimizer, T_max=300, eta_min=1e-6),
            "interval": "epoch",
            "frequency": 1
        }
        return [optimizer], [scheduler]


# Multi-scale sampler and data loading logic can be implemented separately
