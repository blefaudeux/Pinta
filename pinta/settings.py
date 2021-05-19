from enum import Enum
from pathlib import Path
from typing import Dict, Optional
from dataclasses import dataclass, field
import torch
from typing import List, Tuple
import logging
from serde import deserialize, serialize
from serde.json import to_json, from_json

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
    SGD = "sgd"
    ADAM_W = "adam_w"


@serialize
@deserialize
@dataclass
class DilatedConvSettings:
    dropout: float = 0.9


@serialize
@deserialize
@dataclass
class MLPSettings:
    inner_layers: int = 3


@serialize
@deserialize
@dataclass
class RNNSettings:
    gru_layers: int = 2
    kernel_sizes: List[int] = field(default_factory=lambda: [3, 3])


@serialize
@deserialize
@dataclass
class ConvSettings:
    kernel_sizes: List[int] = field(default_factory=lambda: [3, 3])


@serialize
@deserialize
@dataclass
class TrunkSettings:
    model_type: ModelType = ModelType.DILATED_CONV
    hidden_size: int = 256
    seq_length: int = 81
    embedding_dimensions: int = 16
    conv_width: List[int] = field(default_factory=lambda: [3, 3, 3, 3])
    dilated_conv: DilatedConvSettings = DilatedConvSettings()
    mlp: MLPSettings = MLPSettings()
    rnn: RNNSettings = RNNSettings()
    conv: ConvSettings = ConvSettings()


@serialize
@deserialize
@dataclass
class EncoderSettings:
    hidden_size: int = 32
    feature_size: int = 8


@serialize
@deserialize
@dataclass
class MixerSettings:
    hidden_size: int = 32
    hidden_layers: int = 2


@serialize
@deserialize
@dataclass
class DataSettings:
    train_batch_size: int = 20000
    test_batch_size: int = 1000
    training_ratio: float = 0.9
    shuffle: bool = True
    train_workers: int = 4
    test_workers: int = 1
    statistics: Optional[List[float]] = None


@serialize
@deserialize
@dataclass
class OptimSettings:
    name: Optimizer = Optimizer.ADAM_W
    learning_rate: float = 0.01
    scheduler: Scheduler = Scheduler.REDUCE_PLATEAU
    scheduler_patience: int = 20
    scheduler_factor: float = 0.8
    momentum: float = 0.9


@serialize
@deserialize
@dataclass
class TrainingSettings:
    epoch: int = 1
    optim: OptimSettings = OptimSettings()
    mixed_precision: bool = False


@serialize
@deserialize
@dataclass
class Settings:
    inputs: List[str] = field(default_factory=lambda: ["tws", "twa_x", "twa_y", "helm"])
    tuning_inputs: List[str] = field(default_factory=list)
    outputs: List[str] = field(default_factory=lambda: ["sog"])
    tokens: Dict[str, Dict[str, int]] = field(default_factory=dict)
    transforms: List[Tuple[str, List[int]]] = field(default_factory=list)
    trunk: TrunkSettings = TrunkSettings()
    encoder: EncoderSettings = EncoderSettings()
    mixer: MixerSettings = MixerSettings()
    data: DataSettings = DataSettings()
    training: TrainingSettings = TrainingSettings()
    log: str = "pinta"


def get_name(params: Settings) -> str:
    _amp = _amp_available and device.type == torch.device("cuda").type and params.training.mixed_precision

    return (
        params.trunk.model_type
        + "_seq_"
        + str(params.trunk.seq_length)
        + "_hidden_"
        + str(params.trunk.hidden_size)
        + "_batch_"
        + str(params.data.train_batch_size)
        + "_lr_"
        + str(params.training.optim.learning_rate)
        + "_ep_"
        + str(params.training.epoch)
        + "_amp_"
        + str(_amp)
    )


def save(filename, settings: Settings):
    filepath = Path(filename).absolute()
    with open(filepath, "w") as file:
        file.write(to_json(settings))


def load(filename) -> Settings:
    filepath = Path(filename).absolute()
    print(f"opening {filepath}")
    with open(filepath, "r") as file:
        return from_json(Settings, file.read())
