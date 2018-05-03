import sys
import csv
import numpy as np

# print("writing...")
text = open(sys.argv[2], 'r', encoding='big5') 
# text = open('../../ml/test_case.csv', 'r', encoding='big5') 
filename = './try_PCA300.csv'
s = open(sys.argv[3], "w+")
s.write("ID,Ans\n")
answer = np.load("map.npy")

row = csv.reader(text , delimiter=",")
for i, r in enumerate(row):
	if i != 0:
		# print(r[1], r[2])
		p1 = answer[int(r[1])]
		p2 = answer[int(r[2])]
		if p1==p2:	pred = 1
		else:		pred = 0
		# print(p1,p2,pred)
		s.write("{},{}\n".format(i-1, pred))
text.close()