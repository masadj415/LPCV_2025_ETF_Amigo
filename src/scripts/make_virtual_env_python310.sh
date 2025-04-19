#!/bin/bash -x

## Script that creates a virtual environment with Python 3.10
# Intended for use on Ubuntu operating system
# Python 3.10 is required to run the quantization notebook

sudo apt update
# sudo apt install python 
sudo apt install software-properties-common -y
sudo add-apt-repository ppa:deadsnakes/ppa -y
sudo apt update
sudo apt install python3.10 -y
sudo apt install python3.10-venv -y
python3.10 --version

python3.10 -m venv .venv3.10
source .venv3.10/bin/activate
pip install --upgrade pip
pip3 install torch==2.6.0 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu126
pip install -r requirements.txt

