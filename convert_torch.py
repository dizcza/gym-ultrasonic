from pathlib import Path

import keras
import numpy as np
import torch
import torch.nn as nn
from keras.models import load_model
from numpy.testing import assert_array_almost_equal


def convert_keras_model(keras_model):
    torch_layers = []
    for keras_layer in keras_model.layers:
        if isinstance(keras_layer, keras.layers.Dense):
            linear = nn.Linear(keras_layer.input_shape[1], keras_layer.output_shape[1], bias=keras_layer.use_bias)
            weight, bias = keras_layer.get_weights()
            state_dict = dict(weight=torch.from_numpy(weight.T), bias=torch.from_numpy(bias))
            linear.load_state_dict(state_dict)
            torch_layers.append(linear)
            if keras_layer.activation is keras.activations.relu:
                torch_layers.append(nn.ReLU(inplace=True))
            elif keras_layer.activation is keras.activations.tanh:
                torch_layers.append(nn.Tanh())
            elif keras_layer.activation is not keras.activations.linear:
                # linear is identity function by default
                raise ValueError(f"Unknown activation func: '{keras_layer.activation}'")
    actor = nn.Sequential(*torch_layers)
    actor.eval()
    for param in actor.parameters():
        param.requires_grad_(False)
    return actor


def verify(keras_model, torch_model, batch_size=1000):
    input_shape = list(keras_model.input_shape)  # (None, 1, 2)
    input_shape[0] = batch_size  # (batch_size, 1, 2)
    input_array = np.random.random_sample(input_shape).astype(np.float32)
    k = keras_model.predict(np.copy(input_array))  # (100, 2)
    input_tensor = torch.from_numpy(np.copy(input_array))[:, 0, :]  # (100, 2)
    t = torch_model(input_tensor).numpy()
    assert_array_almost_equal(k, t)


model_path = Path('weights/actor.h5')
keras_model = load_model(str(model_path))
actor = convert_keras_model(keras_model)

model_path = model_path.with_suffix('.pt')
torch.save(actor, model_path)
actor = torch.load(model_path)

print(actor)
verify(keras_model, actor)

