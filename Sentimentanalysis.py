from keras.datasets import imdb
from keras import utils
import keras
import tensorflow as tf
import os
import numpy as np

VOCAB_SIZE = 88584

MAXLEN = 250
BATCH_SIZE = 64

(train_data, train_labels), (test_data, test_labels) = imdb.load_data(num_words = VOCAB_SIZE)

train_data = utils.pad_sequences(train_data, MAXLEN)
test_data = utils.pad_sequences(test_data, MAXLEN)

model = tf.keras.Sequential([tf.keras.layers.Embedding(VOCAB_SIZE, 32), tf.keras.layers.LSTM(32), tf.keras.layers.Dense(1, activation="sigmoid")])

model.compile(loss="binary_crossentropy", optimizer="rmsprop", metrics=['acc'])
history = model.fit(train_data, train_labels, epochs = 10, validation_split=0.2)

results = model.evaluate(test_data, test_labels)
print(results)

word_index = imdb.get_word_index()

def encode_text(text):
    tokens = keras.preprocessing.text.text_to_word_sequence(text)
    tokens = [word_index[word] if word in word_index else 0 for word in tokens]
    return utils.pad_sequences([tokens], MAXLEN)[0]

def predict(text):
    encoded_text = encode_text(text)
    pred = np.zeros((1,250))
    pred[0] = encoded_text
    result = model.predict(pred)
    print(result[0])

predict("That was a great movie")
predict("That was a really bad movie")