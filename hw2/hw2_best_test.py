import sys
import csv
import numpy as np
import pandas as pd
import math
def add_feature(DF):
	tmp = DF[['age', 'fnlwgt', 'capital_gain', 'capital_loss', 'hours_per_week']]
	print(tmp.shape)
	DF, tmp = np.asarray(DF), np.asarray(tmp)
	print(DF.shape)
	DF = np.concatenate(
		(DF, tmp**2, tmp**3, tmp**4, tmp**5,
			tmp**6, tmp**7, tmp**8, tmp**9, tmp**10,
			tmp**11, tmp**12, tmp**13, tmp**14, tmp**15,
			tmp**16, tmp**17, tmp**18, tmp**19, tmp**20,
			tmp**21, tmp**22, tmp**23, tmp**24, tmp**25,
			tmp**26, tmp**27, tmp**28, tmp**29, tmp**30,
			tmp**31, tmp**32, tmp**33, tmp**34, tmp**35,
			tmp**36, tmp**37, tmp**38, tmp**39, tmp**40,
			tmp**41, tmp**42, tmp**43, tmp**44, tmp**45,
			tmp**46, tmp**47, tmp**48, tmp**49, tmp**50,
			np.sin(tmp), np.cos(tmp), np.tan(tmp), np.arctan(tmp)),
		axis = 1)
	return DF

x = pd.read_csv(sys.argv[1])
# print(x.columns)
x = add_feature(x)
y = pd.read_csv(sys.argv[2], header = None)
y = np.ravel(y)
X_test = pd.read_csv(sys.argv[3])
X_test = add_feature(X_test)

X_train, Y_train = x, y

from sklearn.preprocessing import StandardScaler
sc=StandardScaler()
sc.fit(X_train)
X_train = sc.transform(X_train)
X_test = sc.transform(X_test)


from sklearn.externals import joblib
lr = joblib.load("train_model.m")


ans = []
a = lr.predict(X_test)
for i in range(len(X_test)):
	ans.append([str(i+1)])
	ans[i].append(a[i])
filename = sys.argv[4]
text = open(filename, "w+")
s = csv.writer(text, delimiter=',', lineterminator='\n')
s.writerow(["id","label"])
for i in range(len(ans)):
	s.writerow(ans[i]) 
text.close()