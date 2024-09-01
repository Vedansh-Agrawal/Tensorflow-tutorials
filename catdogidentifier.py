import os
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras

import tensorflow_datasets as tfds
tfds.disable_progress_bar()

IMG_SIZE = 160

def format_example(image,label):
    image = tf.cast(image, tf.float32)
    image = (image/127.5) - 1
    image = tf.image.resize(image, (IMG_SIZE, IMG_SIZE))
    return image, label

(raw_train, raw_validation, raw_test), metadata = tfds.load('cats_vs_dogs', split = ['train[:80%]', 'train[80%:90%]', 'train[90%:]'], with_info = True, as_supervised = True,)

get_label_name = metadata.features['label'].int2str

#for image, label in raw_train.take(2):
#    plt.figure()
#    plt.imshow(image)
#    plt.title(get_label_name(label))

train = raw_train.map(format_example)
validation = raw_validation.map(format_example)
test = raw_test.map(format_example)

#for image, label in train.take(2):
#    plt.figure()
#    plt.imshow(image)
#    plt.title(get_label_name(label))


IMG_SHAPE = (IMG_SIZE, IMG_SIZE, 3)

base_model = tf.keras.applications.MobileNetV2(input_shape = IMG_SHAPE, include_top = False, weights = 'imagenet')
#base_model.summary()

base_model.trainable = False

global_average_layer = tf.keras.layers.GlobalAveragePooling2D()
prediction_layer = keras.layers.Dense(1)
model = tf.keras.Sequential([base_model, global_average_layer, prediction_layer])

base_learning_rate = 0.0001
model.compile(optimizer = tf.keras.optimizers.RMSprop(lr = base_learning_rate), loss = tf.keras.losses.BinaryCrossentropy(from_logits = True), metrics = ['accuracy'])

initial_epochs = 3
validation_steps = 20

loss0,accuracy0 = model.evaluate(validation, steps = validation_steps)