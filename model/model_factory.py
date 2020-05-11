from datetime import datetime
from typing import Any, Dict

import settings
from model.model_cnn import Conv
from model.model_dilated_conv import TemporalModel
from model.model_mlp import Mlp
from settings import ModelType


def model_factory(params: Dict[str, Any], model_path: str):

    log_directory = "logs/" + settings.get_name() + "_" + str(datetime.now())

    if params["model_type"] == ModelType.Conv:
        INPUT_SIZE = [len(params["inputs"]), params["seq_length"]]

        dnn = Conv(
            logdir=log_directory,
            input_size=INPUT_SIZE,
            hidden_size=params["hidden_size"],
            kernel_size=params["conv_width"],
            output_size=len(params["outputs"]),
            filename=model_path,
        )

    if params["model_type"] == ModelType.DilatedConv:
        dnn = TemporalModel(
            logdir=log_directory,
            num_input_channels=len(params["inputs"]),
            num_output_channels=len(params["outputs"]),
            filter_widths=params["conv_width"],
            dropout=0.25,
            channels=params["hidden_size"],
            filename=model_path,
        )

    if params["model_type"] == ModelType.MultiLayerPerceptron:
        dnn = Mlp(
            logdir=log_directory,
            input_size=len(params["inputs"]),
            hidden_size=params["hidden_size"],
            number_hidden_layers=params["mlp_inner_layers"],
            output_size=len(params["outputs"]),
            filename=model_path,
        )

    dnn.to(settings.device)
    return dnn
