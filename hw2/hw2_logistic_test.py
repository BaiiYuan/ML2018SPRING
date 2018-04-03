import sys
import csv 
import math
import numpy as np
w = np.load("model_logistic.npy")
# test
test_x = []
n_row = 0
text = open(sys.argv[1] ,"r")
row = csv.reader(text , delimiter= ",")
for r in row:
	if n_row != 0:
		r = [ float(i) for i in r ]
		test_x.append(r)
	n_row = n_row+1
text.close()
test_x = np.array(test_x)
test_x1 = test_x[:, 0:10]
test_x2 = test_x[:, 11:]
test_x3 = test_x[:, 78:81]
test_x = np.concatenate((test_x1, test_x2), axis=1)

test_x = np.concatenate( ( np.ones((test_x.shape[0],1)),test_x ) , axis=1)
ans = []
for i in range(len(test_x)):
	ans.append([str(i+1)])
	a = np.sign(np.dot(w,test_x[i]))
	if a == -1:
		a = 0
	else:
		a = 1
	ans[i].append(a)

filename = sys.argv[2]
text = open(filename, "w+")
s = csv.writer(text, delimiter=',', lineterminator='\n')
s.writerow(["id","label"])
for i in range(len(ans)):
	s.writerow(ans[i]) 
text.close()