import datetime
import torch
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.profilers import PyTorchProfiler
import json
from pytorch_lightning.loggers import TensorBoardLogger

# Import your dataset modules
import dataset.utils as dsutils
import utils.compute_class_weights as compute_class_weights
import os
from datetime import datetime
from pytorch_lightning import Callback
from training.lightning_model import LightningModel

class SaveConfigCallback(Callback):
    def __init__(self, config: dict, model_name: str, file_name: str):
        """
        Args:
            config (dict): Your config dictionary.
            model_name (str): Name of the model (used for folder name).
            file_name (str): Name of the JSON file to save.
        """
        self.config = config
        self.model_name = model_name
        self.file_name = file_name
        self.config_saved = False

    def on_train_epoch_end(self, trainer, pl_module):
        if trainer.current_epoch == 0 and not self.config_saved:
            save_dir = os.path.join("saved_configs", self.model_name)
            os.makedirs(save_dir, exist_ok=True)
            
            # Add current timestamp to the filename
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            base_name, ext = os.path.splitext(self.file_name)
            new_file_name = f"{base_name}_{timestamp}{ext}"
            
            save_path = os.path.join(save_dir, new_file_name)

            # Save the config file
            with open(save_path, "w") as f:
                json.dump(self.config, f, indent=4)
            
            print(f"✅ Config saved at {save_path}")
            self.config_saved = True

def get_lightning_model(pytorch_model, config_file_path, dataloader = None):

    """
    Function to create a LightningModel instance with the given PyTorch model and configuration file.
    """

    with open(config_file_path, "r") as f:
        training_parameters = json.load(f)
        training_config = dsutils.TrainingConfig(**training_parameters)
    
    if training_config.balance_dataset is True and dataloader != None:
        weights = compute_class_weights.get_class_weights(dataloader, recompute=True)
    else:
        weights = None

    if training_config.balance_datasets_per_class is True and dataloader != None:
        table = compute_class_weights.get_class_dataset_weights(dataloader, recompute=True)
    else:
        table = None

    model = LightningModel(
        pytorch_model, 
        learning_rate=training_config.learning_rate, 
        dataset_weights=training_config.dataset_weights,
        class_weights=weights,
        table=table,
        label_smoothing = training_config.label_smoothing,
        per_class_dataset_weights = training_config.use_per_class_dataset_weighthing,
        weight_decay = training_config.weight_decay,
    )

    return model


def lightning_train(config_file_path , pytorch_model):
    """
    Function to train a PyTorch model using PyTorch Lightning.
    Args:
        config_file_path (str): Path to the configuration file.
        pytorch_model (nn.Module): The PyTorch model to be trained.
    """
    # Define device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    with open(config_file_path, "r") as f:
        training_parameters = json.load(f)
        training_config = dsutils.TrainingConfig(**training_parameters)

    num_datasets = len(training_config.dataset_weights)



    #--------------------------
    # Define the training datasets
    #--------------------------

    train_datasets = []

    if(num_datasets >= 1):
        dataset_imagenet_train = dsutils.get_imagenet_dataset(
            config = training_config.imagenet_config_train,
            mode = "train"
        )
        train_datasets.append(dataset_imagenet_train)
        print("Len Imagenet train", len(dataset_imagenet_train))

    if(num_datasets >= 2):
        using_coco_new = True # True if using the new COCO dataset format

        if(using_coco_new):
            dataset_coco_train = dsutils.get_new_coco_dataset(
                config = training_config.SD_config_train, mode = "train"
                )
        else:
            dataset_coco_train = dsutils.get_coco_dataset(
                config = training_config.coco_config_train,
                mode = "train",
            )   
        train_datasets.append(dataset_coco_train)
        print("Len Coco train", len(dataset_coco_train))

    if(num_datasets >= 3):
        dataset_stable_diffusion_train = dsutils.get_stable_diffusion_dataset(
            config = training_config.SD_config_train,
            mode = "train"
        )
        train_datasets.append(dataset_stable_diffusion_train)
        print("Len SD train", len(dataset_stable_diffusion_train))

    if(num_datasets >= 4):
        dataset_styles_train = dsutils.get_styles_dataset(
            config = training_config.styles_config_train,
            mode = "train"
        )
        train_datasets.append(dataset_styles_train)
        print("Len Styles train", len(dataset_styles_train))

    train_dataset = torch.utils.data.ConcatDataset(train_datasets)

    #--------------------------
    # Define the validation datasets
    #--------------------------

    dataset_imagenet_val = dsutils.get_imagenet_dataset(
        config = training_config.imagenet_config_val,
        mode = "val"
    )

    print("Len imagenet val", len(dataset_imagenet_val))

    if(num_datasets >= 2):
        using_coco_new_val = True # True if using the new COCO dataset format

        if(using_coco_new_val):
            dataset_coco_val = dsutils.get_new_coco_dataset(
                config = training_config.SD_config_train, mode = "val"
                )
        else:
            dataset_coco_val = dsutils.get_coco_dataset(
                config = training_config.coco_config_val,
                mode = "val"
            )

    print("Len coco val", len(dataset_coco_val))


    dataset_stable_diffusion_val = dsutils.get_stable_diffusion_dataset(
        config = training_config.SD_config_val,
        mode = "val"
    )

    print("Len SD val", len(dataset_stable_diffusion_val))

    val_dataset = torch.utils.data.ConcatDataset([dataset_coco_val, dataset_imagenet_val, dataset_stable_diffusion_val])
    
    # Define DataLoaders
    train_loader = dsutils.get_dataloader(dataset = train_dataset, config = training_config.dataloader_config, shuffle=True)
    val_loader = dsutils.get_dataloader(dataset = val_dataset, config = training_config.dataloader_config, shuffle=False)

    # Define the model name and version
    # This is used to save the model and logs in a specific folder structure
    model_name    = training_config.model_configuration.model_name
    model_version = training_config.model_configuration.version

    dir_extension = model_name + "/" + "Ver" + model_version


    # Saves the model every epoch
    checkpoint_callback= ModelCheckpoint(
        dirpath="models/" + dir_extension,
        filename= model_name + "-{epoch:02d}-{train_loss:.4f}-{val_loss:.4f}",
        save_top_k=50,
        monitor="val_loss",
        mode="min"
    )

    # Defines the tensorboard logger and the pytorch profiler
    logger = TensorBoardLogger("logs/" + dir_extension, name=model_name, default_hp_metric=False)
    profiler = PyTorchProfiler(on_trace_ready=torch.profiler.tensorboard_trace_handler("tb_logs/" + dir_extension),
                            trace_memory=True,
                            schedule=torch.profiler.schedule(skip_first=10, warmup=1, wait=1, active=20)
                            )

    # Save the training configuration to a JSON file
    # This is done using the SaveConfigCallback class
    # at the end of the first epoch
    save_config_callback = SaveConfigCallback(
        config=training_parameters,
        model_name=model_name,
        file_name="training_config.json"
    )

    # Create the Lightning model
    # This is a wrapper around the PyTorch model
    model = get_lightning_model(
        pytorch_model, 
        config_file_path,
        dataloader = train_loader
    )

    # Define the trainer
    # This is the main class that handles the training loop
    trainer = Trainer(
        profiler=profiler,
        max_epochs=training_config.max_epochs,
        devices= "auto",
        accelerator="auto",
        strategy="auto",
        precision="16-mixed",
        callbacks=[checkpoint_callback, save_config_callback],
        logger=logger,
    )

    # Sanity check to ensure that the model is on the correct device
    print(trainer.device_ids)

    # Train the model
    trainer.fit(model,  train_dataloaders=train_loader, val_dataloaders=val_loader)
