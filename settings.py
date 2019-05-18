import json

# Our lightweight data structure..
from collections import namedtuple
Dataframe = namedtuple("Dataframe", ["input", "output"])


_DEFAULTS = {
    "inputs": ['wind_speed', 'wind_angle_x', 'wind_angle_y', 'rudder_angle'],
    "outputs": ['boat_speed'],
    "network_filename": "trained/conv.pt",
    "hidden_size": 128,
    "seq_length": 64,
    "training_ratio": 0.9,
    "batch_size": 15000,
    "epoch": 1
}


def get_defaults():
    return _DEFAULTS


def save(filename, settings):
    with open(filename, "w") as file:
        json.dump(settings, file)


def load(filename):
    with open(filename, "r") as file:
        return json.load(file)
