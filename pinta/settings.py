import json
from enum import Enum
from pathlib import Path
from typing import Any, Dict
from dataclasses import dataclass, field
import torch
from typing import List, Tuple
import logging
from typed_json_dataclass import TypedJsonMixin

_amp_available = hasattr(torch.cuda, "amp") and hasattr(torch.cuda.amp, "autocast")

# Select our target at runtime
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")


if torch.cuda.is_available():
    logging.warning("CUDA enabled")
else:
    logging.warning("CPU enabled")


class ModelType(str, Enum):
    CONV = "conv"
    DILATED_CONV = "dilated_conv"
    MLP = "mlp"
    RNN = "rnn"


class Scheduler(str, Enum):
    COSINE = "cos"
    REDUCE_PLATEAU = "reduce_plateau"


class Optimizer(str, Enum):
    SGD = "SGD"
    ADAM_W = "ADAM_W"


@dataclass
class TrunkSettings(TypedJsonMixin):
    model_type: ModelType = ModelType.DILATED_CONV
    hidden_size: int = 256
    conv_width: List[int] = field(default_factory=lambda: [3, 3, 3, 3])


@dataclass
class DataSettings(TypedJsonMixin):
    train_batch_size: int = 20000
    test_batch_size: int = 1000
    training_ratio: float = 0.9
    shuffle: bool = True
    train_workers: int = 4
    test_workers: int = 1


@dataclass
class OptimSettings(TypedJsonMixin):
    name: Optimizer = Optimizer.ADAM_W
    learning_rate: float = 0.01
    scheduler: Scheduler = Scheduler.REDUCE_PLATEAU
    scheduler_patience: int = 20
    scheduler_factor: float = 0.8
    momentum: float = 0.9


@dataclass
class TrainingSettings(TypedJsonMixin):
    epoch: int = 1
    optim: OptimSettings = OptimSettings()


@dataclass
class Settings(TypedJsonMixin):
    inputs: List[str] = field(default_factory=lambda: ["tws", "twa_x", "twa_y", "helm"])
    tuning_inputs: List[str] = field(default_factory=list)
    outputs: List[str] = field(default_factory=lambda: ["sog"])
    transforms: List[Tuple[str, List[Any]]] = field(default_factory=list)
    seq_length: int = 81
    trunk: TrunkSettings = TrunkSettings()
    data: DataSettings = DataSettings()
    training: TrainingSettings = TrainingSettings()


def get_name(params: Dict[str, Any]):
    _amp = _amp_available and device.type == torch.device("cuda").type and params["amp"]

    return (
        params["trunk"]["model_type"]
        + "_seq_"
        + str(params["seq_length"])
        + "_hidden_"
        + str(params["trunk"]["hidden_size"])
        + "_batch_"
        + str(params["data"]["train_batch_size"])
        + "_lr_"
        + str(params["optim"]["learning_rate"])
        + "_ep_"
        + str(params["epoch"])
        + "_amp_"
        + str(_amp)
    )


def save(filename, settings: Settings):
    filepath = Path(filename).absolute()
    with open(filepath, "w") as file:
        json.dump(settings.__dict__, file)


def load(filename) -> Settings:
    filepath = Path(filename).absolute()
    print(f"opening {filepath}")
    with open(filepath, "r") as file:
        return json.load(file)
