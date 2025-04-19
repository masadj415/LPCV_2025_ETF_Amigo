import datetime
import torch
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.profilers import PyTorchProfiler
import json
from pytorch_lightning.loggers import TensorBoardLogger
import torchsummary

# Import your dataset modules
import dataset.utils as dsutils
import utils.compute_class_weights as compute_class_weights
import os
from datetime import datetime
from pytorch_lightning import Callback
from training.lightning_model import LightningModel
from training.from_scratch_lightning_model import LightningModelScratch

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
        per_class_dataset_weights = training_config.balance_datasets_per_class

    )

    return model


def lightning_train(config_file_path , pytorch_model):
    # Define device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    with open(config_file_path, "r") as f:
        training_parameters = json.load(f)
        training_config = dsutils.TrainingConfig(**training_parameters)

    num_datasets = len(training_config.dataset_weights)



    # Load training datasets

    train_datasets = []

    if(num_datasets >= 1):
        dataset_imagenet_train = dsutils.get_imagenet_dataset(
            config = training_config.imagenet_config_train,
            mode = "train"
        )
        train_datasets.append(dataset_imagenet_train)
        print("Len Imagenet train", len(dataset_imagenet_train))

    if(num_datasets >= 2):
        using_coco_new = True

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
    # Validation Dataset
    #--------------------------

    dataset_imagenet_val = dsutils.get_imagenet_dataset(
        config = training_config.imagenet_config_val,
        mode = "val"
    )

    print("Len imagenet val", len(dataset_imagenet_val))

    if(num_datasets >= 2):
        using_coco_new_val = True

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

    model_name    = training_config.model_configuration.model_name
    model_version = training_config.model_configuration.version

    def get_current_time_formatted():
        return datetime.datetime.now().strftime("%y/%m/%d %H:%M:%S")

    dir_extension = model_name + "/" + "Ver" + model_version

    # Define model checkpoint callback
    checkpoint_callback= ModelCheckpoint(
        dirpath="models/" + dir_extension,
        filename= model_name + "-{epoch:02d}-{train_loss:.4f}-{val_loss:.4f}",
        save_top_k=50,
        monitor="val_loss",
        mode="min"
    )

    logger = TensorBoardLogger("logs/" + dir_extension, name=model_name, default_hp_metric=False)
    profiler = PyTorchProfiler(on_trace_ready=torch.profiler.tensorboard_trace_handler("tb_logs/" + dir_extension),
                            trace_memory=True,
                            schedule=torch.profiler.schedule(skip_first=10, warmup=1, wait=1, active=20)
                            )

    # Define callback to save training config
    save_config_callback = SaveConfigCallback(
        config=training_parameters,
        model_name=model_name,
        file_name="training_config.json"
    )

    # Initialize model
    model = get_lightning_model(
        pytorch_model, 
        config_file_path,
        dataloader = train_loader
    )

    # Initialize Trainer with both training and validation loaders
    trainer = Trainer(
        profiler=profiler,
        max_epochs=training_config.max_epochs,
        devices= "auto",  # Use only one GPU or CPU
        accelerator="auto",
        strategy="auto",  # Allows later multi-GPU setup without changing code
        precision="16-mixed",
        callbacks=[checkpoint_callback, save_config_callback],
        logger=logger,
        # overfit_batches=10,
    )

    print(trainer.device_ids)


    trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=val_loader)


import optuna
from casvit.rcvit import rcvit_t

def objective(trial):
    config_file_path = "/home/centar15-desktop1/LPCV_2025_T1/configs/default_training_config.json"

    with open(config_file_path, "r") as f:
        training_parameters = json.load(f)
        
    training_config = dsutils.TrainingConfig(**training_parameters)
            
    alpha = [
        trial.suggest_float("alpha1", 0.5, 1.0),  # Prvi dataset
        trial.suggest_float("alpha2", 0.5, 1.0),  # Drugi dataset
        trial.suggest_float("alpha3", 0.5, 1.0),  # Treći dataset
        trial.suggest_float("alpha4", 0.0, 0.5),  # Četvrti dataset
    ]
    
    dirichlet_distribution = torch.distributions.Dirichlet(torch.tensor(alpha))
      
    # Train the model with new dataset weights
    pytorch_model = rcvit_t()  # Ensure this is your model class
    model = get_lightning_model(pytorch_model, config_file_path)
            
    trainer = Trainer(
                max_epochs=5,  # Reduce epochs for faster optimization
                devices="auto",
                accelerator="auto",
                precision="16-mixed",
            )

    trainer.fit(model)

    # Retrieve validation loss for Optuna to minimize
    val_loss = trainer.callback_metrics["val_loss"].item()

    return val_loss

# Create an Optuna study and optimize
study = optuna.create_study(direction="minimize")
study.optimize(objective, n_trials=20)

print(f"Best parameters: {study.best_params}")