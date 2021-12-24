#! /bin/bash

# Upgrade python
sudo add-apt-repository ppa:deadsnakes/ppa

sudo apt --assume-yes update
sudo apt --assume-yes install software-properties-common
sudo apt --assume-yes install python3.9
sudo apt --assume-yes install python3-pip

# Set up a python 3.9 environment
cd /home/david
virtualenv --python=/usr/bin/python3.9 py39
source py39/bin/activate

# Clone the repository
cd /home/david
git clone https://github.com/davidnorman314/CEO.git

# Update python libraries
cd /home/david/CEO/src
pip install -r requirements.txt

# Test. Note that the test takes a few seconds, since it writes the full
# Q table results to disk.
cd /home/david/CEO/src
pytest
python -m learning.qlearning --episodes 100

