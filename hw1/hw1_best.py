import sys
import csv 
import math
import random
import numpy as np
import random

IDlist = [9]
Hours = 9
Limit = 130
w = np.load("model_hw1_best.npy")
# w = [ 1.63813038e+00, -1.00978456e-01, 1.80654144e-01, 1.95491318e-01,
# -2.25686612e-01, 2.88118959e-02, 3.79189662e-01, -4.84650648e-01,
# 2.58619333e-01, 6.72988239e-01, 1.74321385e-03, -8.82849307e-03,
# -2.28043400e-03, -1.57490752e-03, -1.97005672e-03, 7.91053071e-03,
# -6.72185119e-03, -9.65300694e-03, 2.16272536e-02, 2.76801666e-05
# , 1.61736759e-04, 1.12243821e-04, 3.07200455e-05, 3.68894760e-05,
# -1.35042216e-04, 1.12441145e-04, 1.78353663e-04, -5.07236910e-04 ,
# -5.75273758e-07, -1.78279631e-06, -1.33653206e-06, 1.06638640e-07,
# -1.18083421e-06, 1.31155710e-06, 6.57942392e-08, -3.35515096e-06,
# 6.74899264e-06, 2.16424524e-09, 7.83581730e-09, 5.08117146e-09,
# -2.66279127e-09, 8.28394090e-09, -5.31249446e-09, -5.54166535e-09,
# 2.02623463e-08, -3.23369499e-08 ]
w = np.asarray(w)
# print(w)
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
test_x = np.concatenate((test_x,test_x**2,test_x**3,test_x**4,test_x**5), axis=1)
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