import numpy as np
import torch
import torch.nn as nn

from model.engine import NN


class Conv(NN):
    """
    Two layers of convolutions, then fully connected, fairly naive but baseline
    """

    def __init__(
        self, logdir, log_channel, input_size, hidden_size, kernel_size, filename=None
    ):
        super(Conv, self).__init__(logdir, log_channel)

        # ----
        # Define the model
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = 1

        # Conv front end
        # First conv is a depthfwise convolution
        # Remark: All inputs convolved to all outputs. This could be changed
        # with the groups flag
        self.conv = nn.Sequential(
            nn.Conv1d(input_size[0], hidden_size, kernel_size=kernel_size),
            nn.LeakyReLU(),
            nn.Conv1d(hidden_size, hidden_size, kernel_size=kernel_size),
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

        # Load from trained NN if required
        try:
            if filename is not None and self.load(filename):
                self._valid = True
                return
        except RuntimeError:
            pass

        self.log.warning(
            "Could not load the specified net," " needs to be computed from scratch"
        )

    def get_layer_weights(self):
        return [layer.weight for layer in self.conv]

    def load(self, filename):
        try:
            with open(filename, "rb") as f:
                self.load_state_dict(torch.load(f))
                self.log.info("---\nNetwork {} loaded".format(filename))
                self.log.info(self)
                return True

        except (ValueError, OSError, IOError, TypeError) as exception:
            self.log.warning(exception)
            self.log.warning("Could not find or load existing NN")
            return False

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
