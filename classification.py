from __future__ import absolute_import, division, print_function, unicode_literals
import matplotlib

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from IPython.display import clear_output

import tensorflow as tf

def input_fn(features, labels, training = True, batch_size = 256):
    ds = tf.data.Dataset.from_tensor_slices((dict(features),labels))

    if training:
        ds = ds.shuffle(1000).repeat()
    return ds.batch(batch_size)



CSV_COLUMN_NAMES = ["SepalLength","SepalWidth","PetalLength","PetalWidth","Species"]
SPECIES = ["Setosa","Versicolor","Virginica"]

train_path = tf.keras.utils.get_file("iris_training.csv","https://storage.googleapis.com/download.tensorflow.org/data/iris_training.csv")
test_path = tf.keras.utils.get_file("iris_test.csv","https://storage.googleapis.com/download.tensorflow.org/data/iris_test.csv")
train = pd.read_csv(train_path,names = CSV_COLUMN_NAMES, header = 0)
test = pd.read_csv(test_path, names = CSV_COLUMN_NAMES, header = 0)

#print(train.head())

train_y = train.pop("Species")
test_y = test.pop("Species")

feature_columns = []
for key in train.keys():
    feature_columns.append(tf.feature_column.numeric_column(key=key))

classifier = tf.estimator.DNNClassifier(feature_columns = feature_columns, hidden_units=[30, 10], n_classes=3)

classifier.train(input_fn = lambda: input_fn(train, train_y), steps = 5000)
result = classifier.evaluate(input_fn = lambda: input_fn(test, test_y, training = False))

#print(result["accuracy"])

# To predict the class of flower based on user input
def inp(features, batch_size = 256):
    return tf.data.Dataset.from_tensor_slices(dict(features)).batch(batch_size)

features = ["SepalLength","SepalWidth","PetalLength","PetalWidth"]
predict = {}

print("Give input")
for feature in features:
    valid = True
    while valid:
        val = input(feature + ": ")
        if not val.isdigit(): valid = False

    predict[feature] = [float(val)]

prediction = classifier.predict(input_fn = lambda: inp(predict))

for pred_dict in prediction:
    class_id = pred_dict['class_ids'][0]
    probability = pred_dict['probabilities'][class_id]

print('prediction is "{}"({:.1f}%)'.format(SPECIES[class_id], 100* probability))

