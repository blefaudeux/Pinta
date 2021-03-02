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
from pinta.model.encoder import TuningEncoder
from pinta.model.mixer import Mixer


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
    model_type = ModelType(params["trunk"]["model_type"])  # will assert if unknown type
    input_size = [len(params["inputs"]), params["seq_length"]]

    # This architecture requires a fixed sequence of convolutions
    # depending on the sequence length
    conv_widths = {
        SequenceLength.small: [3, 3, 3],
        SequenceLength.medium: [3, 3, 3, 3],
        SequenceLength.large: [3, 3, 3, 3, 3],
    }[SequenceLength(params["seq_length"])]

    # Hack to make sure that the model are not actually instantiated in the dict
    def lazy(x):
        def getter():
            return x

        return getter

    # Use a big look up table to get the proper model out for the trunk
    # One benefit of that is that the settings become strict, anything which
    # does not match a key will cause an assert

    trunk_outputs = (
        params["encoder"]["embedding_dimensions"]
        if "embedding_dimensions" in params["encoder"].keys()
        else len(params["outputs"])
    )

    trunk = {
        ModelType.DILATED_CONV: lazy(
            TemporalModel(
                logdir=log_directory,
                num_input_channels=len(params["inputs"]),
                num_output_channels=trunk_outputs,
                filter_widths=conv_widths,
                dropout=params["trunk"]["dilated_conv"]["dropout"],
                channels=params["trunk"]["hidden_size"],
                filename=model_path,
            )
        ),
        ModelType.CONV: lazy(
            Conv(
                logdir=log_directory,
                input_size=input_size,
                hidden_size=params["trunk"]["hidden_size"],
                kernel_sizes=params["trunk"]["conv"]["kernel_sizes"],
                output_size=trunk_outputs,
                filename=model_path,
            )
        ),
        ModelType.RNN: lazy(
            ConvRNN(
                logdir=log_directory,
                input_size=len(params["inputs"]),
                hidden_size=params["trunk"]["hidden_size"],
                kernel_sizes=params["trunk"]["rnn"]["kernel_sizes"],
                n_gru_layers=params["trunk"]["rnn"]["gru_layers"],
                output_size=trunk_outputs,
                filename=model_path,
            )
        ),
        ModelType.MLP: lazy(
            Mlp(
                logdir=log_directory,
                input_size=len(params["inputs"]),
                hidden_size=params["trunk"]["hidden_size"],
                number_hidden_layers=params["trunk"]["mlp"]["inner_layers"],
                output_size=trunk_outputs,
                filename=model_path,
            )
        ),
    }[model_type]()

    logging.info(f"Model used for the trunk:\n{trunk}")

    # By default we only have the trunk
    model = trunk

    # Now optionally build an encoder for the tuning inputs
    if "tuning_inputs" in params.keys() and len(params["tuning_inputs"]) > 0:
        tuning_encoder = TuningEncoder(
            inputs=len(params["tuning_inputs"]),
            hidden=params["encoder"]["hidden_size"],
            out_features=params["encoder"]["feature_size"],
        )

        # Given the original trunk and the encoder, the signal needs to be mixed
        # and have some downstream capacity
        model = Mixer(logdir=log_directory, trunk=trunk, tuning_encoder=tuning_encoder, params=params)

    return model.to(settings.device)
