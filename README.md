# Hydra Examples

This repository contains a basic introductory example on how to use the [Hydra](https://hydra.cc/) framework for configuring and running computational experiments. The objective of this framework is to facilitate running and collecting results of experiments with multiple variations or combinations of configuration parameters.

### Basic example: Measuring the running time of several NN architectures

The `original_script.py` file contains a basic, pure Python implementation of our use case: Create a set of different neural network architectures and measure its execution time. As you can see, the network layer sizes are hardcoded, and the script just creates the models and runs them a number of times. Then, it outputs the average and standard deviation running times.

```python
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
```

However, if we want to try different architectures, or execute only some of them, we need to manually adjust the `last_layer_width`, `intermediate_layers_width` and `intermediate_layers_depth` and re-run the script. This gives us poor flexibility.

#### Adapting the use case to [Hydra](https://hydra.cc/)

The `hydrified_script.py` and `config.yaml` files contain the implementation of the very same use case, but adapted to Hydra. Let's start by taking a look at `config.yaml`:

```yaml
model: Model1

Model1:
    layers_size: [1, 315, 2]

Model2:
    layers_size: [2, 315, 2]

Model5:
    layers_size: [5, 315, 2]

Model10:
    layers_size: [10, 310, 2]

Model20:
    layers_size: [20, 180, 4]

Model50:
    layers_size: [50, 175, 4]

Model100:
    layers_size: [100, 200, 3]

Model200:
    layers_size: [200, 100, 9]

Model500:
    layers_size: [500, 100, 6]

Model1000:
    layers_size: [1000, 100, 1]
```

It contains the definition of each model separately, with a name and a list of three numbers corresponding to the the `last_layer_width`, `intermediate_layers_width` and `intermediate_layers_depth` variables. It also defines a configuration key, `model`, that by default is assigned to `Model1`. Then, we have the adapted Python script, `hydrified_script.py`, as follows:

```python
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
```

The code looks basically the same, with just one decorator to adapt it to Hydra and inject the configuration options through the `config` variable. Then, we assume a single neural network architecture, we create its model and measure its execution time exactly the same way as before.

Now, if we run this script (`$python hydrified_script.py`) we would get the execution time of just `Model1`, since it is the default value of the `model` configuration key. However, we can conveniently run any other model with the command line, for example:

`$ python hydrified_script.py model=Model50` would execute only `Model50`.

And we can also run any combination of models with the `--multirun` flag:

`$ python hydrified_script.py --multirun model=Model1,Model100,Model1000`

### Further reading

This is just a shot to start getting familiar with Hydra, as the options it offers are endless. We strongly recommend you to follow the [tutorials](https://hydra.cc/docs/tutorials/intro/) and check the [common usage patterns](https://hydra.cc/docs/intro/). In particular, these two are most suitable for machine learning experiments:

- [Specializing Configuration](https://hydra.cc/docs/patterns/specializing_config/).
- [Configuring Experiments](https://hydra.cc/docs/patterns/configuring_experiments/).

