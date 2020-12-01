#!/bin/bash

# Adopted from Brown CS1470 Assignment

# stop if any command failed
set -e 

# this installs the virtualenv module
python3 -m pip install virtualenv
# this creates a virtual environment named "env"
python3 -m venv env
# this activates the created virtual environment
source env/bin/activate
# updates pip
python3 -m pip install -U pip
# install setup tools
python3 -m pip install setuptools==41.2.0
# this installs the required python packages to the virtual environment
python3 -m pip install -r requirements.txt

echo created environment