#! /bin/bash

# Upgrade python
sudo add-apt-repository ppa:deadsnakes/ppa
sudo apt install python3.9

# Install conda
sudo apt-get install tcsh
curl https://repo.anaconda.com/archive/Anaconda3-2021.05-Linux-x86_64.sh > CondaInstall.sh

# Install conda. Have the installer run conda init.
/bin/sh CondaInstall.sh

conda create -n py39 python=3.9

conda activate py39

conda install -c conda-forge gym

# Other libs
conda install matplotlib


# Try to install stable-baselines3
# Upgrade glibc
# sudo apt-get install libc6
# conda install -c conda-forge stable-baselines3

# Install blobfuse for Azure blob storage access
wget https://packages.microsoft.com/config/ubuntu/20.04/packages-microsoft-prod.deb
sudo dpkg -i packages-microsoft-prod.deb
sudo apt-get update
sudo apt-get install blobfuse
