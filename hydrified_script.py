#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 19 14:51:44 2024

@author: T. Teijeiro
"""

import hydra
import keras
import timeit
import numpy as np

@hydra.main(version_base=None, config_path=".", config_name="config")
def main(config):
    input_shape = (1,)
    x = keras.random.normal(input_shape)
    model = [keras.layers.Input(shape=input_shape)]
    #Read model parameters from the config
    last_layer_width, intermediate_layers_width, intermediate_layers_depth = config[config.model].layers_size
    for k in range(intermediate_layers_depth):
        model.append(keras.layers.Dense(intermediate_layers_width, activation="tanh"))
    model.append(keras.layers.Dense(last_layer_width, activation="tanh"))
    model = keras.Sequential(model)
    model.name = config.model
    model.compile()
    model.summary()
    meas = timeit.repeat(lambda:model(x), number=1000)
    #Since there are 1000 repetitions, and measures are in seconds, the final
    #units are already in msec
    print(f'Timing: {np.mean(meas):.4f} ms ± {np.std(meas):.4f} ms per loop')

if __name__ == "__main__":
    main()
