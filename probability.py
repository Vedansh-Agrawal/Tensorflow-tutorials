from __future__ import absolute_import, division, print_function, unicode_literals
import matplotlib

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from IPython.display import clear_output

import tensorflow as tf
import tensorflow_probability as tfp

tfd = tfp.distributions
initial_distribution = tfd.Categorical(probs=[0.8, 0.2])
transition_distribution = tfd.Categorical(probs = [[0.7, 0.3],[0.2, 0.8]])

observation_distribution = tfd.Normal(loc = [0., 15.], scale = [5., 10.])

model = tfd.HiddenMarkovModel(initial_distribution = initial_distribution, transition_distribution = transition_distribution, observation_distribution = observation_distribution, num_steps = 7)
 
mean = model.mean()

with tf.compat.v1.Session() as sess:
    print(mean.numpy())