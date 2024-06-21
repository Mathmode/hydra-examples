#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 19 14:51:44 2024

@author: T. Teijeiro
"""

import keras
import timeit
import numpy as np

if __name__ == "__main__":
    # Architecture of the collection of neural networks: 1 -> m -> m -> ... -> m -> N, where m repeats k times
    # The input is 1-dimensional and the output is N-dimensional. We have m-dimensional hidden layers.
    # We create 10 models
    last_layer_width = [1, 2, 5, 10, 20, 50, 100, 200, 500, 1000] # value of N
    intermediate_layers_width = [315, 315, 315, 310, 180, 175, 200, 100, 100, 100] # value of m
    intermediate_layers_depth = [2,2,2,2,4,4,3,9,6,1] # value of k

    input_shape = (1,)
    x = keras.random.normal(input_shape)

    models = []
    for i in range(len(last_layer_width)):
      model = [keras.layers.Input(shape=input_shape)]
      for k in range(intermediate_layers_depth[i]):
        model.append(keras.layers.Dense(intermediate_layers_width[i], activation="tanh"))
      model.append(keras.layers.Dense(last_layer_width[i], activation="tanh"))
      models.append(keras.Sequential(model))
      models[i].name = f"Model{last_layer_width[i]}"
      models[i].compile()

    for model in models:
        model.summary()
        meas = timeit.repeat(lambda:model(x), number=1000)
        #Since there are 1000 repetitions, and measures are in seconds, the final
        #units are already in msec
        print(f'Timing: {np.mean(meas):.4f} ms ± {np.std(meas):.4f} ms per loop\n')
