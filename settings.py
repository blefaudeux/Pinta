import json

# Our lightweight data structure..
from collections import namedtuple
Dataframe = namedtuple("Dataframe", ["input", "output"])


_DEFAULTS = {
    "inputs": ['wind_speed', 'wind_angle_x', 'wind_angle_y', 'rudder_angle'],
    "outputs": ['boat_speed'],
    "network_filename": "trained/conv.pt",
    "hidden_size": 64,
    "seq_length": 128,
    "training_ratio": 0.9,
    "batch_size": 8000,
    "epoch": 50,
    "dataset_normalization": {
        "mean": [13.60, -0.08, 0.28, -0.57],
        "std": [2.41, 0.32, 0.52, 7.22]
    }
}


def get_defaults():
    return _DEFAULTS


def save(filename, settings):
    with open(filename, "w") as file:
        json.dump(settings, file)


def load(filename):
    with open(filename, "r") as file:
        return json.load(file)
