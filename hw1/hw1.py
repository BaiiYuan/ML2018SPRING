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

w = np.load("model_hw1.npy")
print(w)
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