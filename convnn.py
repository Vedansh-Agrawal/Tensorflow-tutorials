import tensorflow as tf
from tensorflow import keras


import numpy as np
import matplotlib.pyplot as plt

(train_images, train_labels), (test_images, test_labels) = keras.datasets.cifar10.load_data()

train_images = train_images / 255.0
test_images = test_images / 255.0

class_names = ["airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"]

#plt.imshow(train_images[5], cmap=plt.cm.binary)
#plt.xlabel(class_names[train_labels[5][0]])
#plt.show()

model = keras.models.Sequential()
model.add(keras.layers.Conv2D(32,(3,3),activation='relu',input_shape=(32,32,3)))
model.add(keras.layers.MaxPooling2D((2,2)))
model.add(keras.layers.Conv2D(64,(3,3),activation='relu'))
model.add(keras.layers.MaxPooling2D((2,2)))
model.add(keras.layers.Conv2D(64,(3,3),activation='relu'))

model.add(keras.layers.Flatten())
model.add(keras.layers.Dense(64,activation='relu'))
model.add(keras.layers.Dense(10))

#model.summary()

model.compile(optimizer='adam',loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True), metrics=['accuracy'])
history = model.fit(train_images, train_labels, epochs=10,validation_data=(test_images,test_labels))

test_loss, test_acc = model.evaluate(test_images, test_labels, verbose = 2)
print(test_acc)
