import json
from enum import Enum
from typing import Any, Dict

import torch

# Select our target at runtime
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
dtype = torch.float32


if torch.cuda.is_available():
    print("CUDA enabled")
else:
    print("CPU enabled")


class ModelType(str, Enum):
    CONV = "conv"
    DILATED_CONV = "dilated_conv"
    MLP = "mlp"
    RNN = "rnn"


# dilated conv:
# sequences: 27,81,243
# filters [3, x3/4/5]

_DEFAULTS = {
    "inputs": ["wind_speed", "wind_angle_x", "wind_angle_y", "rudder_angle"],
    "outputs": ["boat_speed"],
    "model_type": ModelType.RNN,
    "hidden_size": 128,
    "seq_length": 27,
    "conv_width": [3, 3, 3],
    "training_ratio": 0.9,
    "train_batch_size": 22000,
    "val_batch_size": 1000,
    "epoch": 2,
    "learning_rate": 1e-1,
    "batch_norm_momentum": 0.1,
    "mlp_inner_layers": 2,  # MLP Specific
    "rnn_gru_layers": 2,  # RNN Specific
    "conv_dilated_dropout": 0.25,  # Dilated conv specific
    "log": "pinta",
}

assert isinstance(_DEFAULTS["model_type"], ModelType), "Unkonwn model type"


def get_default_params():
    return _DEFAULTS


def get_name(params: Dict[str, Any] = _DEFAULTS):
    return (
        params["model_type"]
        + "_seq_"
        + str(params["seq_length"])
        + "_hidden_"
        + str(params["hidden_size"])
        + "_batch_"
        + str(params["train_batch_size"])
        + "_lr_"
        + str(params["learning_rate"])
        + "_ep_ "
        + str(params["epoch"])
        + "_bnm_"
        + str(params["batch_norm_momentum"])
    )


def save(filename, settings):
    with open(filename, "w") as file:
        json.dump(settings, file)


def load(filename):
    with open(filename, "r") as file:
        return json.load(file)
