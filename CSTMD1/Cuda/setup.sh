#!/bin/bash
# Do a clean up to make sure all changes propagate into the setup
./cleanup.sh
# load up cuda
source /vol/cuda/7.5.18/setup.sh
# do the setup
python setup.py install --user
