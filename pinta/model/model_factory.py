import logging
from datetime import datetime
from enum import Enum

import pinta.settings as settings
from pinta.model.model_base import NN
from pinta.model.cnn import Conv
from pinta.model.dilated_conv import TemporalModel
from pinta.model.mlp import Mlp
from pinta.model.rnn import ConvRNN
from pinta.settings import ModelType
from pinta.model.encoder import TuningEncoder
from pinta.model.mixer import Mixer
from pinta.model.transformer import Transformer
from omegaconf import DictConfig
import torch


class SequenceLength(int, Enum):
    "supported sequence lengths for the dilated convolutions architecture"

    small = 27
    medium = 81
    large = 243


def model_factory(params: DictConfig, model_path: str) -> NN:
    """Given pipeline params, generate the appropriate model.

    Args:
        params (Dict[str, Any]):  General pipeline parameters
        model_path (str): Optional, path of a saved model

    Returns:
        NN : inference model
    """

    log_directory = "logs/" + model_path.replace(".pt", "") + "_" + str(datetime.now())
    model_type = params.model.model_type
    input_size = [len(params.inputs), params.model.seq_length]

    # This architecture requires a fixed sequence of convolutions
    # depending on the sequence length
    conv_widths = {
        SequenceLength.small: [3, 3, 3],
        SequenceLength.medium: [3, 3, 3, 3],
        SequenceLength.large: [3, 3, 3, 3, 3],
    }[SequenceLength(params.model.seq_length)]

    # Hack to make sure that the model are not actually instantiated in the dict
    def lazy(Constructor, args):
        def getter():
            return Constructor(**args)

        return getter

    # Use a big look up table to get the proper model out for the trunk
    # One benefit of that is that the settings become strict, anything which
    # does not match a key will cause an assert

    trunk_outputs = (
        params.model.embedding_dimensions
        if len(params.tuning_inputs) > 0
        else len(params.outputs)
    )

    if model_type == ModelType.DILATED_CONV:
        trunk: torch.nn.Module = TemporalModel(
            logdir=log_directory,
            num_input_channels=len(params.inputs),
            num_output_channels=trunk_outputs,
            filter_widths=conv_widths,
            dropout=params.model.dilated_conv.dropout,
            channels=params.model.hidden_size,
            filename=model_path,
        )
    elif model_type == ModelType.CONV:
        trunk = Conv(
            logdir=log_directory,
            input_size=input_size,
            hidden_size=params.model.hidden_size,
            kernel_sizes=params.model.kernel_sizes,
            output_size=trunk_outputs,
            filename=model_path,
        )
    elif model_type == ModelType.RNN:
        trunk = ConvRNN(
            logdir=log_directory,
            input_size=len(params.inputs),
            hidden_size=params.model.hidden_size,
            kernel_sizes=params.model.kernel_sizes,
            n_gru_layers=params.model.gru_layers,
            output_size=trunk_outputs,
            filename=model_path,
        )
    elif model_type == ModelType.MLP:
        trunk = Mlp(
            logdir=log_directory,
            input_size=len(params.inputs),
            hidden_size=params.model.hidden_size,
            number_hidden_layers=params.model.inner_layers,
            output_size=trunk_outputs,
            filename=model_path,
        )
    elif model_type == ModelType.TRANSFORMER:
        trunk = Transformer(
            logdir=log_directory,
            input_size=len(params.inputs),
            hidden_size=params.model.hidden_size,
            output_size=trunk_outputs,
            filename=model_path,
        )
    else:
        raise ValueError(f"Model type {model_type} not supported")

    logging.info(f"Model used for the trunk:\n{trunk}")

    # By default we only have the trunk
    model = trunk

    # Now optionally build an encoder for the tuning inputs
    if len(params.tuning_inputs) > 0:
        tuning_encoder = TuningEncoder(
            inputs=len(params.tuning_inputs),
            hidden=params.encoder.hidden_size,
            out_features=params.encoder.feature_size,
        )

        # Given the original trunk and the encoder, the signal needs to be mixed
        # and have some downstream capacity
        model = Mixer(
            logdir=log_directory,
            trunk=trunk,
            tuning_encoder=tuning_encoder,
            params=params,
        )
        logging.info("Using a final Mixer to get the predictions")

    return model.to(settings.device)
