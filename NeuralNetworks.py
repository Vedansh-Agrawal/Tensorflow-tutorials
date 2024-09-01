import tensorflow as tf
from tensorflow import keras


import numpy as np
import matplotlib.pyplot as plt

fashion_mnist = keras.datasets.fashion_mnist

(train_images, train_labels),(test_images, test_labels) = fashion_mnist.load_data()

#print(train_images.shape)

class_names = ["T-shirt/top", "Trouser", "Pullover", "Dress", "Coat", "Sandal", "Shirt", "Sneaker", "Bag","Ankle boot"]

#plt.figure()
#plt.imshow(test_images[10])
#plt.colorbar()
#plt.grid(False)
#plt.show()

train_images = train_images / 255.0 #Data Preprocess to make the values between zero or one
test_images = test_images / 255.0

model = keras.Sequential([keras.layers.Flatten(input_shape=(28, 28)), keras.layers.Dense(128, activation='relu'),keras.layers.Dense(10,activation='softmax')])
#Relu is Rectified Linear unit
#softmax makes sure the output is a probability distribution

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

model.fit(train_images, train_labels, epochs=10) #training process

test_loss, test_acc = model.evaluate(test_images, test_labels, verbose = 1)
#print(test_acc)
#this gives lesser accuracy cause of overfitting

predictions = model.predict(test_images)
#print(predictions[0]) #this is aprobabilities of all the classes
#print(class_names[np.argmax(predictions[300])]) #returns index of maximum probability
#plt.figure()
#plt.imshow(test_images[300])
#plt.colorbar()
#plt.grid(False)
#plt.show()