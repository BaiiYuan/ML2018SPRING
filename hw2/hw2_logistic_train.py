import sys
import csv 
import math
import numpy as np

def theeeeeeeta(s):
	# print(s)
	return 1.0 / (1 + math.e**(-s))
# from sklearn.model_selection import train_test_split
# X_train, X_test, Y_train, Y_test = train_test_split(x, y, test_size = 0.25, random_state = 112)
x = []
y = []
n_row = 0
text = open(sys.argv[1], 'r', encoding='big5') 
row = csv.reader(text , delimiter=",")
for r in row:
	# if n_row == 0:
	# 	for i in range(len(r)):
	# 		print("index = ", i, " ", r[i])
	# 	exit()
	if n_row != 0:
		r = [ float(i) for i in r ]
		x.append(r)
	n_row = n_row+1
text.close()
x = np.asarray(x)
# print(x)
text = open(sys.argv[2], 'r', encoding='big5') 
row = csv.reader(text , delimiter=",")
for r in row:
	tmp = int(r[0])
	if tmp == 0:
		tmp = -1
	y.append(tmp)

text.close()
y = np.asarray(y)
# print(y)

x1 = x[:, 0:10]
x2 = x[:, 11:]
x3 = x[:, 78:81]

x = np.concatenate((x1, x2), axis=1)
x = np.concatenate( ( np.ones((x.shape[0], 1)), x ) , axis=1)
# print(len(x))

lr = 1e-2		# learning rate
Iter = 5000			# iteration
NUM = len(x)
x_t = x.transpose()
s_gra = np.zeros(len(x[0]))
# w = np.zeros(len(x[0]))	# initial weight vector
w = np.dot(np.linalg.pinv(x),y)
# w = np.load("model_hw2.npy")
s_gra = np.zeros(len(x[0]))
Lambda = 1e-6

# tmp_ans = np.sign(np.dot(x, w))
# tmp_cou = 0
# for k in range(NUM):
# 	if tmp_ans[k] == y[k]:
# 		tmp_cou+=1
# print('epoch: %d | w length %f | equal rate: %f' % (0, np.sum(w**2), tmp_cou/NUM) )

for i in range(Iter):
	tmp = np.dot(x.T, -y*theeeeeeeta(-y*np.dot(x, w)))
	gra = tmp/NUM + Lambda * w
	s_gra += gra**2
	ada = np.sqrt(s_gra)
	w -= lr* gra/ada
	lr = lr*0.999
	if i%50 == 49:
		tmp_ans = np.sign(np.dot(x, w))
		tmp_cou = 0
		for k in range(NUM):
			if tmp_ans[k] == y[k]:
				tmp_cou+=1
		print('epoch: %d | w length %f | equal rate: %f' % (i+1, np.sum(w**2), tmp_cou/NUM) )
		# print(np.log(np.ones((x.shape[0], 1))+math.e**(np.dot(-y, np.dot(x, w)))))
			# input()

np.save("model_logistic.npy",w)



