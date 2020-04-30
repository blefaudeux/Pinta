import json

import torch

# Select our target at runtime
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
dtype = torch.float32


if torch.cuda.is_available():
    print("CUDA enabled")
else:
    print("CPU enabled")


_DEFAULTS = {
    "inputs": ["wind_speed", "wind_angle_x", "wind_angle_y", "rudder_angle"],
    "outputs": ["boat_speed"],
    "network_root_name": "conv",
    "hidden_size": 128,
    "seq_length": 64,
    "training_ratio": 0.9,
    "train_batch_size": 16000,
    "val_batch_size": 1000,
    "epoch": 200,
    "log": "pinta",
}


def get_defaults():
    return _DEFAULTS


def get_name():
    return (
        _DEFAULTS["network_root_name"]
        + "_seq_"
        + str(_DEFAULTS["seq_length"])
        + "_hidden_"
        + str(_DEFAULTS["hidden_size"])
        + "_batch_"
        + str(_DEFAULTS["train_batch_size"])
    )


def save(filename, settings):
    with open(filename, "w") as file:
        json.dump(settings, file)


def load(filename):
    with open(filename, "r") as file:
        return json.load(file)
