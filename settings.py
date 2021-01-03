import json
from enum import Enum
from pathlib import Path
from typing import Any, Dict

import torch

try:
    from torch.cuda.amp import autocast

    _amp_available = True
except ImportError:
    _amp_available = False

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


class Scheduler(str, Enum):
    COSINE = "cos"
    REDUCE_PLATEAU = "reduce_plateau"


# Reference keys handled by the conversion and data processing utils
_EXPAND_KEYS = {
    "tws": "true wind speed",
    "twa": "true wind angle",
    "awa": "apparent wind angle",
    "sog": "speed over ground",
    "cog": "course over ground",
    "twd": "true wind direction",
    "heel": "boat heel angle",
    "trim": "trim angle",
    "az": "azimuth",
    "foil_port_fo_load": "",
    "foil_stbd_fo_load": "",
    "foil_port_rake": "",
    "foil_stbd_rake": "",
    "foil_port_ext": "",
    "foil_stbd_ext": "",
    "bobstay_load": "",
    "j2_load": "",
    "outrigger_port_load": "outrigger_port_load",
    "outrigger_stbd_load": "outrigger_stbd_load",
    "runner_port_load": "runner_port_load",
    "runner_stbd_load": "runner_stbd_load",
    "lat": "latitude",
    "lon": "longitude",
}


_DEFAULTS = {
    "inputs": ["tws", "twa_x", "twa_y", "helm"],
    "outputs": ["sog"],
    "model_type": ModelType.DILATED_CONV,
    "hidden_size": 256,
    "seq_length": 81,
    "conv_width": [3, 3, 3, 3],
    "training_ratio": 0.9,
    "train_batch_size": 10 ** 3,
    "val_batch_size": 10 ** 2,
    "epoch": 30,
    "learning_rate": 5 * 1e-2,
    "dilated_conv": {"dropout": 0.25},
    "mlp": {"inner_layers": 3},
    "rnn": {"gru_layers": 2, "kernel_sizes": [3, 3]},
    "conv": {"kernel_sizes": [3, 3]},
    "log": "pinta",
    "scheduler": Scheduler.REDUCE_PLATEAU,
    "amp": False,
}

assert isinstance(_DEFAULTS["model_type"], ModelType), "Unkonwn model type"


def get_default_params():
    return _DEFAULTS


def get_name(params: Dict[str, Any]):
    _amp = _amp_available and device.type == torch.device("cuda").type and params["amp"]

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
        + "_ep_"
        + str(params["epoch"])
        + "_amp_"
        + str(_amp)
    )


def save(filename, settings):
    filepath = Path(filename).absolute()
    with open(filepath, "w") as file:
        json.dump(settings, file)


def load(filename):
    filepath = Path(filename).absolute()
    print(f"opening {filepath}")
    with open(filepath, "r") as file:
        return json.load(file)
