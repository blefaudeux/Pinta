from typing import List

import numpy as np
import torch
import torch.nn as nn
from pinta.model.model_base import NN


class Conv(NN):
    """
    Two layers of convolutions, then fully connected, fairly naive but baseline
    """

    def __init__(
        self,
        logdir: str,
        input_size: List[int],
        hidden_size: int,
        kernel_sizes: List[int],
        strides: List[int] = [1, 1],
        output_size: int = 1,
        filename=None,
    ):
        super(Conv, self).__init__(logdir, "TemporalConv")

        assert len(kernel_sizes) == 2, "Only two kernel sizes supported"
        assert len(strides) == 2, "Only two strides supported"

        # ----
        # Define the model
        self.hidden_size = hidden_size
        self.output_size = output_size

        # Conv front end
        self.conv = nn.Sequential(
            nn.Conv1d(
                input_size[0],
                hidden_size,
                kernel_size=kernel_sizes[0],
                stride=strides[0],
            ),
            nn.LeakyReLU(),
            nn.Conv1d(
                hidden_size, hidden_size, kernel_size=kernel_sizes[1], stride=strides[1]
            ),
            nn.LeakyReLU(),
        )

        out_conv_size = self._get_conv_out(input_size)
        self.log.info(
            "Feature vector size out of the convolution is {}".format(out_conv_size)
        )

        # Ends with two fully connected layers
        self.fc = nn.Sequential(
            nn.Linear(out_conv_size, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, self.output_size),
        )

        # Load from trained model if required
        try:
            if filename is not None and self.load(filename):
                self._valid = True
                return
        except RuntimeError:
            pass

        self.log.warning(
            "Could not load the specified model," " needs to be computed from scratch"
        )

    def forward(self, inputs, *kwargs):
        # One feature vector per sample in. Rearrange accordingly
        features = self.conv(inputs).view(inputs.size()[0], -1)

        # The feature vector goes through the fully connected layers,
        # and we're good
        return self.fc(features), None

    def _get_conv_out(self, shape):
        # Useful to compute the shape out of the conv blocks
        # (including eventual padding..)
        o = self.conv(torch.zeros(1, *shape))
        return int(np.prod(o.size()))
