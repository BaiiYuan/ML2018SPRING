#!/bin/bash
# bash hw3_train.sh <training data>
wget https://www.dropbox.com/s/csfdoh15a2mefez/BestMix2.h5?dl=1
python3 Train.py $1
