# bash ./hw2_logistic.sh $1 $2 $3 $4 $5 $6
echo start training
# python3 hw2_logistic_train.py $3 $4
python3 hw2_logistic_train.py ../../hw2/train_X ../../hw2/train_Y
echo end training
echo start testing
# python3 hw2_logistic_test.py $5 $6
python3 hw2_logistic_test.py ../../hw2/test_X ./predict/log.csv
echo end testing