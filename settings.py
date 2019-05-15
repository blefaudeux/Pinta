import json

_DEFAULTS = {
    "inputs": ['wind_speed', 'wind_angle_x', 'wind_angle_y', 'rudder_angle'],
    "outputs": ['boat_speed'],
    "network_filename": "trained/conv.pt",
    "hidden_size": 64,
    "seq_length": 256,
    "training_ratio": 0.9,
    "batch_size": 10000,
    "epoch": 20
}


def get_defaults():
    return _DEFAULTS


def save(filename, settings):
    with open(filename, "w") as f:
        json.dump(settings, f)


def load(filename):
    with open(filename, "r") as f:
        return json.load(f)
