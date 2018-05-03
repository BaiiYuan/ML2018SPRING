
import sys
import csv
import pandas as pd
import numpy as np
import os
os.environ["CUDA_VISIBLE_DEVICES"] = '0'
import keras.layers as layers
from keras.utils import * #np_utils, print_summary
from keras.models import Sequential, Model
from keras.layers import Dense, Activation, Convolution2D, MaxPooling2D, Flatten, Dropout, Add, Input, AveragePooling2D, average
from keras.optimizers import Adam, RMSprop
from keras import regularizers




from keras.models import load_model
models=[]
for i in range(4):
	add = './model' + str(i+1) + '.h5'
	modelTemp=load_model(add) # load model
	modelTemp.name="aUniqueModelName"+ str(i+1) # change name to be unique
	models.append(modelTemp)


print('------------ Ensembling ------------')



def ensembleModels(models, model_input):
	yModels=[model(model_input) for model in models] 
	yAvg=layers.average(yModels)
	modelEns = Model(inputs=model_input, outputs=yAvg, name='ensemble')
	# modelEns.compile(optimizer=Adam(lr=1e-4), loss='categorical_crossentropy', metrics=['accuracy'])
	print (modelEns.summary())

	modelEns.save('mix2_model.h5')
	print(type(modelEns))

model_input = Input(shape=models[0].input_shape[1:]) # c*h*w
ensembleModels(models, model_input)

print('------------ Saveing ------------')

X_test = []
text = open('../../ml/test.csv', 'r', encoding='big5') 
# text = open('../../DATA/hw3/test.csv', 'r', encoding='big5') 
row = csv.reader(text , delimiter=",")
for i, r in enumerate(row):
	if i != 0:
		k = np.asarray([float(n) for n in r[1].split()]).reshape((48, 48))
		X_test.append(k)
text.close()
X_test = np.asarray(X_test)
X_test = X_test.reshape(-1, 1,48, 48)/255.
print(len(X_test))

Yeee = load_model('mix2_model.h5')

a = Yeee.predict(X_test)
a = np.argmax(a, axis=1)
ans = []
N = len(X_test)
for i in range(N):
	ans.append([str(i)])
	ans[i].append(a[i])
filename = './final.csv'
text = open(filename, "w+")
s = csv.writer(text, delimiter=',', lineterminator='\n')
s.writerow(["id","label"])
for i in range(len(ans)):
	s.writerow(ans[i]) 
text.close()