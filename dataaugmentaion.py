#This is a neural network for smaller datasets

from keras.preprocessing import image
from keras.preprocessing.image import ImageDataGenerator

import tensorflow as tf
from tensorflow import keras


import numpy as np
import matplotlib.pyplot as plt

(train_images, train_labels), (test_images, test_labels) = keras.datasets.cifar10.load_data()


datagen = ImageDataGenerator(rotation_range = 40, width_shift_range = 0.2, height_shift_range = 0.2, shear_range = 0.2, zoom_range = 0.2, horizontal_flip = True, fill_mode='nearest')

train_images = train_images/255.0

test_img = train_images[3]
img = keras.utils.img_to_array(test_img)
img = img.reshape((1,) + img.shape)

i = 0

for batch in datagen.flow(img, save_prefix = 'test', save_format = 'jpeg'):
    plt.figure(i)
    plot = plt.imshow(batch[0])
    i += 1
    if i > 4:
        break

plt.show()