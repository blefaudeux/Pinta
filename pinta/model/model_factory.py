import logging
from datetime import datetime
from enum import Enum
from typing import Any, Dict

import pinta.settings as settings
from pinta.model.model_base import NN
from pinta.model.model_cnn import Conv
from pinta.model.model_dilated_conv import TemporalModel
from pinta.model.model_mlp import Mlp
from pinta.model.model_rnn import ConvRNN
from pinta.settings import ModelType


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

    log_directory = "logs/" + settings.get_name(params) + "_" + str(datetime.now())
    model_type = ModelType(params["model_type"])  # will assert if unknown type

    def get_conv_model():
        INPUT_SIZE = [len(params["inputs"]), params["seq_length"]]

        return Conv(
            logdir=log_directory,
            input_size=INPUT_SIZE,
            hidden_size=params["hidden_size"],
            kernel_sizes=params["conv"]["kernel_sizes"],
            output_size=len(params["outputs"]),
            filename=model_path,
        )

    def get_dilated_conv_model():
        # This architecture requires a fixed sequence of convolutions
        # depending on the sequence length
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
            dropout=params["dilated_conv"]["dropout"],
            channels=params["hidden_size"],
            filename=model_path,
        )

    def get_mlp_model():
        return Mlp(
            logdir=log_directory,
            input_size=len(params["inputs"]),
            hidden_size=params["hidden_size"],
            number_hidden_layers=params["mlp"]["inner_layers"],
            output_size=len(params["outputs"]),
            filename=model_path,
        )

    def get_rnn_model():
        return ConvRNN(
            logdir=log_directory,
            input_size=len(params["inputs"]),
            hidden_size=params["hidden_size"],
            kernel_sizes=params["rnn"]["kernel_sizes"],
            n_gru_layers=params["rnn"]["gru_layers"],
            output_size=len(params["outputs"]),
            filename=model_path,
        )

    model = {
        ModelType.DILATED_CONV: get_dilated_conv_model,
        ModelType.CONV: get_conv_model,
        ModelType.RNN: get_rnn_model,
        ModelType.MLP: get_mlp_model,
    }[model_type]()

    logging.info(f"Model used:\n{model}")

    model.to(settings.device)
    return model
