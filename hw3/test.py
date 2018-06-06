import sys
import csv
import pandas as pd
import numpy as np



from keras.models import load_model

print('------------ Loading ------------')
Yeee = load_model('BestMix2.h5?dl=1')

print('------------ Saveing ------------')

X_test = []
text = open(sys.argv[1], 'r', encoding='big5') 
# text = open('../../DATA/hw3/test.csv', 'r', encoding='big5') 
row = csv.reader(text , delimiter=",")
for i, r in enumerate(row):
	if i != 0:
		k = np.asarray([float(n) for n in r[1].split()]).reshape((48, 48))
		X_test.append(k)
text.close()
X_test = np.asarray(X_test)
X_test = X_test.reshape(-1, 1,48, 48)/255.

print('------------ Predicting ------------')
a = Yeee.predict(X_test)
a = np.argmax(a, axis=1)
ans = []
N = len(X_test)
for i in range(N):
	ans.append([str(i)])
	ans[i].append(a[i])
filename = sys.argv[2]
text = open(filename, "w+")
s = csv.writer(text, delimiter=',', lineterminator='\n')
s.writerow(["id","label"])
for i in range(len(ans)):
	s.writerow(ans[i]) 
text.close()
