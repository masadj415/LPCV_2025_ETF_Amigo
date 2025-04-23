# LPCV 2025 Track 1 Solution

## Introduction

This repository contains our solution for Track 1 of the LPCV 2025 Challenge, where we secured **3rd place**.
Our team consists of final-year bachelor students from the University of Belgrade, School of Electrical Engineering.
## Challenge Overview

- **Competition Website**: [LPCV 2025](https://lpcv.ai/2025LPCVC/image-classification)
- **Track**: Track 1 - Image Classification for Different Lighting Conditions and Styles
- **Objective**: Develop a lightweight and accurate image classification model suitable for resource-constrained hardware environments
- **Final Accuracy**: **95.12%** 

## Model details

Our solution is based on the CAS-ViT model.

- **Code Repository**: [CAS-ViT GitHub](https://github.com/Tianfang-Zhang/CAS-ViT)  
- **Paper**: [CAS-ViT](https://arxiv.org/abs/2408.03703)  

We adapted and fine-tuned this model to achieve high accuracy under diverse lighting conditions and styles.


## Training Details

- **Framework**: PyTorch + PyTorch Lightning
- **Checkpoint**: [`FINAL_MODEL-epoch=54-train_loss=0.6094-val_loss=0.8717.ckpt`](https://drive.google.com/file/d/1u_UzRuVfSNPCHaOv33xwhqsd_gvmMHfG/view?usp=drive_link) (this is a Pytorch Lightning checkpoint)
   - Add this checkpoint to the `root/finalLightningCheckpoint` folder
- **Hardware setup**: The model was trained on
   - **GPU**: 1x RTX 5090 (32GB)
   - **CPU**: Ryzen 9 9950X
- **Datasets**: 
  - A custom **COCO** variant
  - A filtered **ImageNet-1K** subset
  - A synthetic dataset we generated using **Stable Diffusion**
  - A **Styles** dataset we designed to improve robustness to various visual style
  - **The dataset is expected to be in the following structure**:
   ```
   root/datasets/
   ├── coco_modified/
   │   ├── train/
   │   │   ├── class_1/
   │   │   │   ├── image1.jpg
   |   |       └── ...
   |   ├── val/
   │   │   ├── ...
   ├── imagenet/
   │   ├── train/
   │   │   ├── class_1/
   │   │   │   ├── image1.jpg
   │   │   │   └── ...
   │   ├── val/
   │   │   ├── ...
   ├── stable_diffusion/
   │   ├── train/
   │   │   ├── class_1/
   │   │   │   ├── image1.jpg
   │   │   │   └── ...
   │   ├── val/
   │   │   ├── ...
   ├── styles/
   │   ├── train/
   │   │   ├── class_1/
   │   │   │   ├── image1.jpg
   │   │   │   └── ...
   ```

- **Hyperparameters**
    - Most hyperparameters are given in the `root/configs/default_training_config.json` file
    - As the `use_per_class_dataset_weighting` parameter is set to `True`, the `dataset_weights` parameter from the `.json` file is not taken into consideration, and instead each class has a different weight depending on which dataset it comes from. These parameters can be found in the `root/src/dataset/weight_class_per_dataset.py` file.


## Setup Instructions

To reproduce our environment and train or evaluate the model, follow these steps:

1. **Create and activate virtual environment** (tested on Python 3.12)  
   `python -m venv .venv`  
   `source .venv/bin/activate`  &nbsp;&nbsp;*(For Windows use: `.\.venv\Scripts\activate`)*

2. **Install the repo in editable mode**  (this also installs the requirements.txt) 
   `pip install -e .`

3. **Install PyTorch and related packages with CUDA support**  
   *(example for CUDA 12.6)*  
   `pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu126`

4. **Download the pretrained weights**  
   from: [CASVIT_T](https://drive.google.com/file/d/1N5Y81Vcyf2ox41TC3wlRBxgQPYaEndTW/view)

5. **Rename the file to** `CASVIT_T.pth` **and put it in the** `root/src/casvit` **folder**

6. **For training, from the root folder run:**  
   `python src/casvit/rcvit.py`

7. **To upload the downloaded checkpoint to QAI Hub, run**
   `python src/casvit/test.py`
   (alternatively change the `pathToLightningCheckpoint` variable in `test.py`)


