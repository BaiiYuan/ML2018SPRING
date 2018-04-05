#!/bin/bash
# bash ./hw2_generative.sh $1 $2 $3 $4 $5 $6
echo start
python3 hw2_generative_train_and_test.py $3 $4 $5 $6 
# python3 hw2_generative_train_and_test.py ../../hw2/train_X ../../hw2/train_Y ../../hw2/test_X ./predict/gen.csv 
echo end