from datetime import datetime
from enum import Enum
from typing import Any, Dict

import settings
from settings import ModelType

from model.model_base import NN
from model.model_cnn import Conv
from model.model_dilated_conv import TemporalModel
from model.model_mlp import Mlp
from model.model_rnn import ConvRNN


class SequenceLength(int, Enum):
    "supported sequence lengths for the dilated convolutions architecture"
    small = 27
    medium = 81
    large = 243


def model_factory(params: Dict[str, Any], model_path: str) -> NN:
    """Given pipeline params, generate the appropriate model.


    Args:
        params (Dict[str, Any]):  General pipeline parameters
        model_path (str): Optional, path of a saved model

    Returns:
        NN : inference model
    """

    log_directory = "logs/" + settings.get_name() + "_" + str(datetime.now())
    model_type = ModelType(params["model_type"])  # will assert if unknown type

    def get_conv_model():
        INPUT_SIZE = [len(params["inputs"]), params["seq_length"]]

        return Conv(
            logdir=log_directory,
            input_size=INPUT_SIZE,
            hidden_size=params["hidden_size"],
            kernel_size=params["conv_width"],
            output_size=len(params["outputs"]),
            filename=model_path,
        )

    def get_dilated_conv_model():
        conv_widths = {
            SequenceLength.small: [3, 3, 3],
            SequenceLength.medium: [3, 3, 3, 3],
            SequenceLength.large: [3, 3, 3, 3, 3],
        }[SequenceLength(params["seq_length"])]

        return TemporalModel(
            logdir=log_directory,
            num_input_channels=len(params["inputs"]),
            num_output_channels=len(params["outputs"]),
            filter_widths=conv_widths,
            dropout=params["conv_dilated_dropout"],
            channels=params["hidden_size"],
            filename=model_path,
        )

    def get_mlp_model():
        return Mlp(
            logdir=log_directory,
            input_size=len(params["inputs"]),
            hidden_size=params["hidden_size"],
            number_hidden_layers=params["mlp_inner_layers"],
            output_size=len(params["outputs"]),
            filename=model_path,
        )

    def get_rnn_model():
        return ConvRNN(
            logdir=log_directory,
            input_size=len(params["inputs"]),
            hidden_size=params["hidden_size"],
            kernel_sizes=params["conv_width"],
            n_gru_layers=params["rnn_gru_layers"],
            output_size=len(params["outputs"]),
            filename=model_path,
        )

    dnn = {
        ModelType.DILATED_CONV: get_dilated_conv_model,
        ModelType.CONV: get_conv_model,
        ModelType.RNN: get_rnn_model,
        ModelType.MLP: get_mlp_model,
    }[model_type]()

    dnn.to(settings.device)
    return dnn
