import sys
import csv 
import math
import random
import numpy as np
import random
data = []
IDlist = [9]#range(18) #
Hours = 9
Limit = 130
for i in range(18):
	data.append([])
n_row = 0
text = open('.train.csv', 'r', encoding='big5') 
row = csv.reader(text , delimiter=",")
for r in row:
	if n_row != 0:
		for i in range(3,27):
			if r[i] != "NR":
				data[(n_row-1)%18].append(float(r[i]))
			else:
				data[(n_row-1)%18].append(float(0))	
	n_row = n_row+1
text.close()
# NUM = len(data[0])



x = []
y = []
for i in range(12):
	for j in range(471):
		x.append([])
		for poison in IDlist:
			for s in range(Hours):
				x[471*i+j].append(data[poison][480*i+j+s] )
		y.append(data[9][480*i+j+9])

T = len(x)
origin_x = x
# print(len(origin_x))
while T > 0:
	for c in range(0*Hours, 1*Hours):
		if x[T-1][c] >= Limit or x[T-1][c] <= 0:
			# print(T)
			del x[T-1]
			del y[T-1]
			break
	T -= 1
x = np.array(x)
y = np.array(y)
# x = np.concatenate((x,x**2,x**3,x**4,x**5), axis=1)
x = np.concatenate((np.ones((x.shape[0], 1)), x), axis=1)
# print(len(x[0]))
w = np.zeros(len(x[0]))	# initial weight vector
lr = 0.005				# learning rate
Iter = 150000			# iteration
Lambda = 1

x_t = x.transpose()
s_gra = np.zeros(len(x[0]))
Sigma = 0

for i in range(Iter):
	
	loss = np.dot(x, w) - y
	cost = np.sum(loss**2) / loss.shape[0]
	gra = np.dot(x_t, loss) + Lambda * w
	s_gra = s_gra*0.9999+gra**2
	ada = np.sqrt(s_gra)
	Sigma += np.sum(gra**2)
	w = w - lr * gra/ada
	if i%10000 == 9999:
		print ('iteration: %d | Cost: %f' % ( i+1, math.sqrt(cost)))

np.save("model_hw1.npy",w)
print("w = ", w)

test_x = []
n_row = 0
text = open(sys.argv[1] ,"r")
row = csv.reader(text , delimiter= ",")

for r in row:
	if n_row %18 == 0:
		test_x.append([])

	if (n_row%18) in IDlist:
		for i in range(2,11):
			if r[i] !="NR":
				test_x[n_row//18].append(float(r[i]))
			else:
				test_x[n_row//18].append(0)
	if n_row%18 == 17:
		index = 0*Hours
		for c in range(Hours-1, -1, -1):
			if test_x[n_row//18][index+c] >= Limit or test_x[n_row//18][index+c] <= 0:
				if c == Hours-1:
					test_x[n_row//18][index+c] = test_x[n_row//18][index+c-2]
				elif c == 0:
					test_x[n_row//18][index+c] = (test_x[n_row//18][index+c+1] + test_x[n_row//18][index+c+2])/2
				else:	
					test_x[n_row//18][index+c] = test_x[n_row//18][index+c+1] 
			
	n_row = n_row+1
text.close()
test_x = np.array(test_x)
# test_x = np.concatenate((test_x,test_x**2,test_x**3,test_x**4,test_x**5), axis=1)
test_x = np.concatenate((np.ones((test_x.shape[0],1)),test_x), axis=1)

ans = []
for i in range(len(test_x)):
	ans.append(["id_"+str(i)])
	a = np.dot(w,test_x[i])
	ans[i].append(a)

filename = sys.argv[2]
text = open(filename, "w+")
s = csv.writer(text, delimiter=',', lineterminator='\n')
s.writerow(["id","value"])
for i in range(len(ans)):
	s.writerow(ans[i]) 
text.close()