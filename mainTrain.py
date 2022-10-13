from ast import Import
from cgi import test
from email.mime import image
import imp
from importlib.resources import path
from statistics import mode
import cv2
import os
import tensorflow as tf
from tensorflow import keras
import numpy as np
from PIL import Image
from sklearn.model_selection import train_test_split
from keras.utils import normalize
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense
from keras.utils import to_categorical
image_directory = 'dataset/'
no_tumer_Images = os.listdir(image_directory+'no/')
yes_tumer_Images = os.listdir(image_directory+'yes/')
dataset = []
lable = []


# print(no_tumer_Images)
# path = 'no0.jpg'
# path.split()

for i, image_name in enumerate(no_tumer_Images):
    if (image_name.split('.')[1] == 'jpg'):
        image = cv2.imread(image_directory+'no/'+image_name)
        image = Image.fromarray(image, 'RGB')
        image = image.resize((64, 64))
        dataset.append(np.array(image))
        lable.append(0)

for i, image_name in enumerate(yes_tumer_Images):
    if (image_name.split('.')[1] == 'jpg'):
        image = cv2.imread(image_directory+'yes/'+image_name)
        image = Image.fromarray(image, 'RGB')
        image = image.resize((64, 64))
        dataset.append(np.array(image))
        lable.append(1)

# print(dataset)
# print(len(lable))
dataset = np.array(dataset)
lable = np.array(lable)

x_train, x_test, y_train, y_test = train_test_split(
    dataset, lable, test_size=0.2, random_state=0)
# print(x_train.shape)
x_train = normalize(x_train, axis=1)
x_test = normalize(x_test, axis=1)

y_train = to_categorical(y_train, num_classes=2)
y_test = to_categorical(y_test, num_classes=2)

# model  Building
model = Sequential()
model.add(Conv2D(32, (3, 3), input_shape=(64, 64, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(32, (3, 3), kernel_initializer="he_uniform"))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(32, (3, 3), kernel_initializer="he_uniform"))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Flatten())
model.add(Dense(64))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(2))
model.add(Activation('softmax'))


model.compile(loss='categorical_crossentropy',
              optimizer='adam', metrics='accuracy')
model.fit(x_train, y_train, batch_size=16, verbose=1, epochs=10,
          validation_data=(x_test, y_test), shuffle=False)

model.save('BrainTumor10EpochsCategorical.h5')
