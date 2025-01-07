from enum import Enum
from pathlib import Path
from typing import Dict, Optional
from dataclasses import dataclass, field
import torch
from typing import List, Tuple, Any
import logging
from serde import deserialize, serialize
from serde.json import to_json, from_json
import omegaconf


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
    TRANSFORMER = "transformer"


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
    dilated_conv: DilatedConvSettings = field(default_factory=DilatedConvSettings)
    mlp: MLPSettings = field(default_factory=MLPSettings)
    rnn: RNNSettings = field(default_factory=RNNSettings)
    conv: ConvSettings = field(default_factory=ConvSettings)


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
    optim: OptimSettings = field(default_factory=OptimSettings)
    bf16: bool = True


@serialize
@deserialize
@dataclass
class Settings:
    inputs: List[str] | omegaconf.listconfig.ListConfig = field(
        default_factory=lambda: ["tws", "twa_x", "twa_y", "helm"]
    )
    tuning_inputs: List[str] | omegaconf.listconfig.ListConfig = field(
        default_factory=list
    )
    outputs: List[str] | omegaconf.listconfig.ListConfig = field(
        default_factory=lambda: ["sog"]
    )
    tokens: Dict[str, Dict[str, int]] = field(default_factory=dict)
    transforms: List[Tuple[str, List[Any]]] | omegaconf.listconfig.ListConfig = field(
        default_factory=list
    )
    trunk: TrunkSettings = field(default_factory=TrunkSettings)
    encoder: EncoderSettings = field(default_factory=EncoderSettings)
    mixer: MixerSettings = field(default_factory=MixerSettings)
    data: DataSettings = field(default_factory=DataSettings)
    training: TrainingSettings = field(default_factory=TrainingSettings)
    log: str = "pinta"

    def __repr__(self):
        use_bf16 = device.type == torch.device("cuda").type and self.training.bf16

        return (
            self.model.model_type
            + "_seq_"
            + str(self.model.seq_length)
            + "_hidden_"
            + str(self.model.hidden_size)
            + "_batch_"
            + str(self.data.train_batch_size)
            + "_lr_"
            + str(self.training.optim.learning_rate)
            + "_ep_"
            + str(self.training.epoch)
            + "_bf16_"
            + str(use_bf16)
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
