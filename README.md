# LPCV 2025 Track 1 Solution

## Introduction

This repository contains our solution for Track 1 of the LPCV 2025 Challenge, where we secured **3rd place**.
## Challenge Overview

- **Competition Website**: [LPCV 2025](https://lpcv.ai/2025LPCVC/image-classification)
- **Track**: Track 1 - Image Classification for Different Lighting Conditions and Styles
- **Objective**: Develop a lightweight and accurate image classification model suitable for resource-constrained hardware environments
- **Final Accuracy**: **95.12%** 


## Training Details

- **Framework**: PyTorch 
- **Checkpoint**: `FINAL_MODEL-epoch=54-train_loss=0.6094-val_loss=0.8717.ckpt`
- **Datasets**: 
  - A custom **COCO** variant
  - A filtered **ImageNet-1K** subset
  - A synthetic dataset we generated using **Stable Diffusion**
  - A **Styles** dataset we designed to improve robustness to various visual styles
  
## Setup Instructions

To reproduce our environment and train or evaluate the model, follow these steps:

1. **Create and activate virtual environment**  
   `python -m venv .venv`  
   `source .venv/bin/activate`  &nbsp;&nbsp;*(For Windows use: `.venv\Scripts\activate`)*

2. **Install the repo in editable mode**  
   `pip install -e .`

3. **Install PyTorch and related packages with CUDA support**  
   *(example for CUDA 12.6)*  
   `pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu126`

4. **Download the pretrained weights**  
   from: [CASVIT_T](https://drive.google.com/file/d/1N5Y81Vcyf2ox41TC3wlRBxgQPYaEndTW/view)

5. **Rename the file to** `CASVIT_T.pth` **and put it in the** `src/casvit` **folder**

6. **For training, from the root folder run:**  
   `python src/casvit/rcvit.py`


