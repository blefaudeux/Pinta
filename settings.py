import json
import torch


# Select our target at runtime
dtype = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor

if dtype == torch.cuda.FloatTensor:
    print("CUDA enabled")
else:
    print("CPU enabled")


_DEFAULTS = {
    "inputs": ['wind_speed', 'wind_angle_x', 'wind_angle_y', 'rudder_angle'],
    "outputs": ['boat_speed'],
    "network_root_name": "conv",
    "hidden_size": 96,
    "seq_length": 64,
    "training_ratio": 0.9,
    "batch_size": 30000,
    "epoch": 10,
    "dataset_normalization": {
        "input": {
            "mean": [13.60, -0.08, 0.28, -0.57],
            "std": [2.41, 0.32, 0.52, 7.22]
        },
        "output": {
            "mean": [6.48],
            "std": [1.69]
        }
    },
    "training": {
        "lr_period_decrease": 20,
        "lr_amount_decrease": 0.9
    }
}


def get_defaults():
    return _DEFAULTS


def get_name():
    return _DEFAULTS["network_root_name"] + "_seq_" + \
        str(_DEFAULTS["seq_length"]) + "_hidden_" + str(_DEFAULTS["hidden_size"]) + \
        "_batch_" + str(_DEFAULTS["batch_size"]) \
        + "_lr_" + str(_DEFAULTS["training"]["lr_period_decrease"]) \
        + "_" + str(_DEFAULTS["training"]["lr_amount_decrease"])


def save(filename, settings):
    with open(filename, "w") as file:
        json.dump(settings, file)


def load(filename):
    with open(filename, "r") as file:
        return json.load(file)
