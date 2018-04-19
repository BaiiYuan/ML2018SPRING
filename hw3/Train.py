import sys
import csv
import pandas as pd
import numpy as np

# np.random.seed(1337)  # for reproducibility
# from keras.datasets import mnist
from keras.utils import * #np_utils, print_summary
from keras.models import Sequential, Model
from keras.layers import Dense, Activation, Convolution2D, MaxPooling2D, Flatten, Dropout, Add, Input, AveragePooling2D
from keras.optimizers import Adam, RMSprop
from keras import regularizers

EPOCH = 300



def Residual(In, height):
	icut = In
	In = Convolution2D(height//4, kernel_size=1, strides=1, padding='same', data_format='channels_first', activation='relu')(In)
	In = Convolution2D(height//4, kernel_size=3, strides=1, padding='same', data_format='channels_first', activation='relu')(In)
	In = Convolution2D(height, kernel_size=1, strides=1, padding='same', data_format='channels_first', activation='relu')(In)
	In = Add()([In, icut])
	In = Activation('relu')(In)
	return In



print("------------ model constructing ------------")
Input = Input(shape=(1, 48, 48))

I = Convolution2D(64, kernel_size=3, strides=2, padding='same', data_format='channels_first', activation='relu')(Input)
I = Residual(I, 64)

icut = Convolution2D(128, kernel_size=1, strides=2, padding='same', data_format='channels_first')(I)
I = Convolution2D(128, kernel_size=3, strides=2, padding='same', data_format='channels_first', activation='relu')(I)
I = Convolution2D(128, kernel_size=3, strides=1, padding='same', data_format='channels_first', activation='relu')(I)
I = Add()([I, icut])
for _ in range(1):
	I = Residual(I, 128)
icut = Convolution2D(256, kernel_size=1, strides=2, padding='same', data_format='channels_first')(I)
I = Convolution2D(256, kernel_size=3, strides=2, padding='same', data_format='channels_first', activation='relu')(I)
I = Convolution2D(256, kernel_size=3, strides=1, padding='same', data_format='channels_first', activation='relu')(I)
I = Add()([I, icut])
for _ in range(1):
	I = Residual(I, 256)

c = AveragePooling2D(pool_size=(2, 2), strides=1, padding='valid', data_format='channels_first')(I)

F = Flatten()(c)
d1 = Dense(1024, activation='relu')(F)
D1 = Dropout(0.3)(d1)
d2 = Dense(256)(D1)
d3 = Dense(7)(d2)
P = Activation('softmax')(d3)

model = Model(inputs=Input, outputs=P)

# Another way to define your optimizer
adam = Adam(lr=1e-4)
# We add metrics to get more results you want to see
model.compile(optimizer=adam,
			  loss='categorical_crossentropy',
			  metrics=['accuracy'])

layer_utils.print_summary(model)



print("------------ data loading ------------")
# data pre-processing
x, y = [], []
text = open(sys.argv[1], 'r', encoding='big5') 
row = csv.reader(text , delimiter=",")
for i, r in enumerate(row):
	if i != 0:
		y.append(int(r[0]))
		k = np.asarray([float(n) for n in r[1].split()]).reshape((48, 48))
		x.append(k)
text.close()
x, y = np.asarray(x), np.asarray(y)
NUM = x.shape[0]
x = x.reshape((NUM, 1, 48, 48)).astype(float)
print(x.shape, y.shape)

## validation
print("------------ validation ------------")
from sklearn import preprocessing, linear_model
from sklearn.model_selection import train_test_split
X_train, X_value, Y_train, Y_value = train_test_split(x, y, test_size = 0.2, random_state = 215)
X_train = X_train.reshape(-1, 1,48, 48)/255.
X_value = X_value.reshape(-1, 1,48, 48)/255.
Y_train = np_utils.to_categorical(Y_train, num_classes=7)
Y_value = np_utils.to_categorical(Y_value, num_classes=7)
print(X_train.shape)
print(Y_train.shape)
print(X_value.shape)
print(Y_value.shape)

# just train
# X_train, Y_train = x, y
# X_train = X_train.reshape(-1, 1, 48, 48)/255.
# Y_train = np_utils.to_categorical(Y_train, num_classes=7)
# print(X_train.shape)
# print(Y_train.shape)

print("------------ ImageDataGenerator ------------")

from keras.preprocessing.image import ImageDataGenerator

datagen = ImageDataGenerator(
	zca_whitening=False,
	rotation_range=30,
	width_shift_range=0.3,
	height_shift_range=0.3,
	shear_range=0.3,
	zoom_range=0.3,
	horizontal_flip=True,
	fill_mode='nearest',
	data_format='channels_first'
)
datagen.fit(X_train)
train_gen = datagen.flow(X_train, Y_train, batch_size = 64)



from keras.callbacks import ReduceLROnPlateau
learning_rate_function = ReduceLROnPlateau(monitor='val_acc', 
											patience=3, 
											verbose=1, 
											factor=0.5, 
											min_lr=1e-12)

print('------------ Training ------------')
# Another way to train the model
# model.fit(X_train, Y_train, epochs=30, batch_size=64, validation_split= 0.2)
train_history = model.fit_generator(train_gen,
					steps_per_epoch=500,
					epochs=EPOCH,
					verbose=1,
					validation_data=(X_value, Y_value)
					# ,callbacks=[learning_rate_function]
								   )



# from keras.models import load_model

print('------------ Saveing ------------')

model.save('./good8_model.h5')  # creates a HDF5 file 'my_model.h5'