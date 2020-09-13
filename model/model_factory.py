from datetime import datetime
from typing import Any, Dict

import settings
from model.model_base import NN
from model.model_cnn import Conv
from model.model_dilated_conv import TemporalModel
from model.model_mlp import Mlp
from model.model_rnn import ConvRNN
from settings import ModelType
import logging


def model_factory(params: Dict[str, Any], model_path: str) -> NN:
    """ Given pipeline params, generate the appropriate model.


    Args:
        params (Dict[str, Any]):  General pipeline parameters
        model_path (str): Optional, path of a saved model

    Returns:
        NN : inference model
    """

    log_directory = "logs/" + settings.get_name() + "_" + str(datetime.now())

    assert isinstance(params["model_type"], ModelType), "Unkonwn model type"

    dnn = None

    if params["model_type"] == ModelType.CONV:
        INPUT_SIZE = [len(params["inputs"]), params["seq_length"]]

        dnn = Conv(
            logdir=log_directory,
            input_size=INPUT_SIZE,
            hidden_size=params["hidden_size"],
            kernel_size=params["conv_width"],
            output_size=len(params["outputs"]),
            filename=model_path,
        )

    if params["model_type"] == ModelType.DILATED_CONV:
        dnn = TemporalModel(
            logdir=log_directory,
            num_input_channels=len(params["inputs"]),
            num_output_channels=len(params["outputs"]),
            filter_widths=params["conv_width"],
            dropout=params["conv_dilated_dropout"],
            channels=params["hidden_size"],
            filename=model_path,
        )

    if params["model_type"] == ModelType.MLP:
        dnn = Mlp(
            logdir=log_directory,
            input_size=len(params["inputs"]),
            hidden_size=params["hidden_size"],
            number_hidden_layers=params["mlp_inner_layers"],
            output_size=len(params["outputs"]),
            filename=model_path,
        )

    if params["model_type"] == ModelType.RNN:
        dnn = ConvRNN(
            logdir=log_directory,
            input_size=len(params["inputs"]),
            hidden_size=params["hidden_size"],
            kernel_sizes=params["conv_width"],
            n_gru_layers=params["rnn_gru_layers"],
            output_size=len(params["outputs"]),
            filename=model_path,
        )

    if dnn is None:
        logging.error(f"Model setting {params["model_type"]} is not supported")
        raise NotImplementedError

    dnn.to(settings.device)
    return dnn
