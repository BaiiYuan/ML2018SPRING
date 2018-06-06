#!/bin/bash
# bash  hw3_test.sh  <testing data>  <prediction file>  <mode>
# mode: public or private
# sed '7,11d' test.py > test2.py; mv test2.py test.py
wget "https://www.dropbox.com/s/csfdoh15a2mefez/BestMix2.h5?dl=1"
# cat BestMix2.h5?dl=1 | sed 's/"amsgrad": false,/                 /g' > BestMix2.h5;
# mv BestMix2.h5 BestMix2.h5?dl=1
# echo error
# cat BestMix2.h5 | grep -a amsgrad
# cat BestMix2.h5 | grep -a beta_2
# echo error

python3 test.py $1 $2
rm -f BestMix2.h5 BestMix2.h5?dl=1	
