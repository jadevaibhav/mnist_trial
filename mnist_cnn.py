import os
from PIL import Image
import numpy as np
path = 'trainingSample'
x = []
y = []
for i in range(10):
    imgs = os.listdir(path+'/'+str(i))
    for j in imgs:
        img = Image.open(path+'/'+str(i)+'/'+j)
        img = np.asarray(img).reshape((28,28,1))
        x.append(img)
        y1 = np.zeros((10))
        y1[i] = 1
        y.append(y1)
x = np.asarray(x)
y = np.asarray(y)
print(x[1].shape)
p = np.random.permutation(600)
x = x[p]
y = y[p]
x_train = x[:550]
y_train = y[:550]
x_val = x[550:600]
y_val = y[550:600]

import tensorflow as tf
import keras
from keras import models
from keras import layers

model = models.Sequential()
model.add(layers.Conv2D(16,kernel_size=(3,3),strides=(1,1),activation='relu',input_shape=(28,28,1)))
model.add(layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
model.add(layers.Conv2D(16,kernel_size=(3,3),strides=(1,1),activation='relu'))
model.add(layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
model.add(layers.Conv2D(32,kernel_size=(3,3),strides=(1,1),activation='relu'))
model.add(layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
model.add(layers.Flatten())
model.add(layers.Dense(100,activation='relu', activity_regularizer=keras.regularizers.l1(0.001)))
model.add(layers.Dense(50,activation='relu', activity_regularizer=keras.regularizers.l1(0.001)))
model.add(layers.Dense(10,activation='softmax', activity_regularizer=keras.regularizers.l1(0.001)))

model.summary()

model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=keras.optimizers.Adam(lr=0.01),
              metrics=['accuracy'])

model.fit(x_train, y_train,
          batch_size=16,
          epochs=50,
          verbose=1,
          validation_data=(x_val, y_val))

model.save('mnist_50.h5')
