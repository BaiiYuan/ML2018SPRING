#!/bin/bash

# bash  hw5_train.sh <training label data>  <training unlabel data>

python3 clean.py $1 $2 ./new_model.h5 Word_test2.h5