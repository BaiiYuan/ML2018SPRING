import sys
import csv 
import math
import numpy as np
import random
x = []
y = []
n_row = 0
text = open(sys.argv[1], 'r', encoding='big5') 
row = csv.reader(text , delimiter=",")
for r in row:
	if n_row != 0:
		r = [ float(i) for i in r ]
		x.append(r)
	n_row = n_row+1
text.close()

x = np.asarray(x)
print(x.shape[0])
text = open(sys.argv[2], 'r', encoding='big5') 
row = csv.reader(text , delimiter=",")
for r in row:
	y.append(int(r[0]))

text.close()
y = np.asarray(y)
print(y)

x1 = x[:, 0:10]
x2 = x[:, 11:]
x = np.concatenate((x1, x2), axis=1)

NUM = x.shape[0]
print(NUM)

x_0, x_1 = [], []
for i in range(NUM):
	if y[i] == 0:
		x_0.append(x[i])
	else:
		x_1.append(x[i])
x_0, x_1 = np.asarray(x_0), np.asarray(x_1)
mu_x0, mu_x1 = np.mean(x_0, axis=0), np.mean(x_1, axis=0)
print(mu_x0.shape)
n_0, n_1, dim = x_0.shape[0], x_1.shape[0], x_0.shape[1]
sig_0, sig_1 = np.cov(x_0.T), np.cov(x_1.T)
# print(sig_0.shape)
sig = (sig_0*n_0+sig_1*n_1)/(n_0+n_1)
# print(np.linalg.pinv(sig))



test_x = []
n_row = 0
text = open(sys.argv[3] ,"r")
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
test_x = np.concatenate((test_x1, test_x2), axis=1)

sig_i = np.linalg.pinv(sig)
ans = []
for i in range(len(test_x)):
	ans.append([str(i+1)])
	a = np.subtract(test_x[i], mu_x0).reshape((-1, 1))
	b = np.subtract(test_x[i], mu_x1).reshape((-1, 1))
	c = np.dot(np.dot(a.T, sig_i), a) - np.dot(np.dot(b.T, sig_i), b)
	c = c.reshape(-1)[0]/2
	# print(c)
	p_0 = 1 / ( 1+n_1/n_0*(math.e**c) )
	print(p_0)
	if p_0 > 0.4 :
		ans[i].append(0)
	else:
		ans[i].append(1)
	# input()
	

filename = sys.argv[4]
text = open(filename, "w+")
s = csv.writer(text, delimiter=',', lineterminator='\n')
s.writerow(["id","label"])
for i in range(len(ans)):
	s.writerow(ans[i]) 
text.close()