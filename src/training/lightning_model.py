import torch
import torch.nn as nn
import torch.optim as optim
import pytorch_lightning as pl
from dataset.weight_class_per_dataset import CLASS_MATRIX

UNIFORM_WEIGHTS = torch.ones([64])
UNIFORM_TABLE   = torch.ones([64, 4])

class LightningModel(pl.LightningModule):
    def __init__(self, model, learning_rate=2e-5, dataset_weights=[0.7, 0.3, 0.0, 0.0], class_weights=UNIFORM_WEIGHTS, weight_decay = 0, table = UNIFORM_TABLE, label_smoothing = 0, per_class_dataset_weights = False):
        super(LightningModel, self).__init__()
        if class_weights is None:
            class_weights = UNIFORM_WEIGHTS
        if table is None:
            table = UNIFORM_TABLE
        
        self.model : nn.Module = model
        self.criterion = nn.CrossEntropyLoss(weight=torch.tensor(class_weights), reduction="none", label_smoothing=label_smoothing)
        self.learning_rate = learning_rate
        print("Init device")
        print(self.device)
        self.dataset_weights = torch.tensor(dataset_weights, requires_grad=False)
        self.table           = torch.tensor(table          , requires_grad=False)

        self.per_class_dataset_weights = per_class_dataset_weights

        self.weight_decay = weight_decay

    def on_fit_start(self):
        self.dataset_weights = self.dataset_weights.to(self.device)
        print("Dataset weights:")
        print(self.dataset_weights)
        self.table           = self.table.to(self.device)
        print("Table per class dataset balancing: ")
        print(self.table)
        print("Weight decay:")
        print(self.weight_decay)
        print("Dataset weights:")
        print(self.dataset_weights)
        print("Using per class dataset weighthing")
        print(self.per_class_dataset_weights)
        if(self.per_class_dataset_weights):
            print(CLASS_MATRIX)
        print(CLASS_MATRIX*self.table)


    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        images, labels, ds = batch  
        # ds = 0 - imagenet, ds = 1 - COCO, ds = 2 - stable diffusion (lightning condions), ds = 3 - different styles
        images, labels = images.to(self.device), labels.to(self.device)
        outputs = self(images)
        loss = self.criterion(outputs, labels)

        ds = ds.to(self.device)

        if(self.per_class_dataset_weights):
            loss = loss*CLASS_MATRIX[labels, ds]
        else:
            loss = loss*self.dataset_weights[ds]

        if self.table is not None:
            loss = loss*self.table[labels, ds]
        
        loss = loss.mean()
    
        self.log("train_loss", loss, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        images, labels, ds = batch
        images, labels = images.to(self.device), labels.to(self.device)
        outputs = self(images)
        loss = self.criterion(outputs, labels)

        ds = ds.to(self.device)

        if(self.per_class_dataset_weights):
            loss = loss*CLASS_MATRIX[labels, ds]
        else:
            loss = loss*self.dataset_weights[ds]

        if self.table is not None:
            loss = loss*self.table[labels, ds]
        
        loss = loss.mean()


        # Calculate accuracy
        preds = torch.argmax(outputs, dim=1)
        acc = (preds == labels).float().mean()

        self.log("val_loss", loss, prog_bar=True)
        self.log("val_acc", acc, prog_bar=True)
        return {"val_loss": loss, "val_acc": acc}


    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)

        return optimizer