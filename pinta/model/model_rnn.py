import logging
from typing import List

import numpy as np
import torch
import torch.nn as nn
from pinta.model.model_base import NN

LOG = logging.getLogger("ConvRNN")


class ConvRNN(NN):
    """
    Combination of a convolutional front end and an RNN (GRU) layer below
     >> see https://gist.github.com/spro/c87cc706625b8a54e604fb1024106556

    """

    def __init__(
        self,
        logdir: str,
        input_size: int,
        hidden_size: int,
        kernel_sizes: List[int],
        n_gru_layers: int,
        output_size: int,
        filename=None,
        tuning_input_size: int = -1,
    ):
        super().__init__(logdir)

        # ----
        # Define the model
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.gru_layers = n_gru_layers

        # Conv front end
        self.conv1 = nn.Conv1d(input_size, hidden_size, kernel_size=kernel_sizes[0])
        self.conv2 = nn.Conv1d(hidden_size, hidden_size, kernel_size=kernel_sizes[1])
        self.relu = nn.ReLU()

        # GRU / LSTM layers
        # Requires [batch, seq, inputs]
        self.gru = nn.GRU(
            hidden_size, hidden_size, n_gru_layers, dropout=0.01, batch_first=True
        )

        # Ends with a fully connected layer
        self.out = nn.Linear(hidden_size, self.output_size)

        # Load from trained NN if required
        if filename is not None:
            self._valid = self.load(filename)
            if self._valid:
                return

            LOG.warning("Could not load the specified net, computing it from scratch")

    def forward(self, inputs, hidden=None):
        # Run through Conv1d and Pool1d layers
        r1 = self.relu(self.conv1(inputs))
        r2 = self.relu(self.conv2(r1))

        # GRU/LSTM layer expects [batch, seq, inputs]
        r2 = r2.transpose(1, 2)
        output_rnn, hidden_out = self.gru(r2, hidden)

        output = self.out(output_rnn[:, -1, :].squeeze())
        return output, hidden_out

    def get_layer_weights(self):
        return self.conv1.weight

    def _get_conv_out(self, shape):
        # Useful to compute the shape out of the conv blocks
        # (including eventual padding..)
        o = self.conv(torch.zeros(1, *shape))
        return int(np.prod(o.size()))
