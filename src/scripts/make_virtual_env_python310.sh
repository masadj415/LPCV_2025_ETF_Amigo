#!/bin/bash -x

## Skripta koja pravi virtuelno okruzenje sa pythonom 3.10
# namenjeno za koriscenje na ubuntu operativnom sistemu
# Python3.10 je potreban za pokretanje sveske za kvantizaciju

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
