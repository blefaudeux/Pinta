import torch
import torch.nn as nn
import numpy as np

from train.engine import NN

dtype = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor


class Conv(NN):
    """
    Pure Conv
    """

    def __init__(self, logdir, input_size, hidden_size, filename=None):
        super(Conv, self).__init__(logdir)

        # ----
        # Define the model
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = 1
        KERNEL_SIZE = 16

        # Conv front end
        # First conv is a depthwise convolution
        self.conv = nn.Sequential(nn.Conv1d(input_size[0], hidden_size,
                                            kernel_size=KERNEL_SIZE),
                                  nn.ReLU(),
                                  nn.Conv1d(hidden_size, hidden_size,
                                            kernel_size=KERNEL_SIZE),
                                  nn.ReLU())

        out_conv_size = self._get_conv_out(input_size)

        # Ends with two fully connected layers
        self.fc = nn.Sequential(nn.Linear(out_conv_size, 512),
                                nn.ReLU(),
                                nn.Linear(512, 256),
                                nn.ReLU(),
                                nn.Linear(256, self.output_size))

        # CUDA switch > Needs to be done after the model has been declared
        if dtype is torch.cuda.FloatTensor:
            print("Using Pytorch CUDA backend")
            self.cuda()

        # Load from trained NN if required
        try:
            if filename is not None and self.load(filename):
                self._valid = True
                return
        except RuntimeError:
            pass

        print("Could not load the specified net, needs to be computed from scratch")

    def load(self, filename):
        try:
            with open(filename, "rb") as f:
                self.load_state_dict(torch.load(f))
                print("---\nNetwork {} loaded".format(filename))
                print(self)
                return True

        except (ValueError, OSError, IOError, TypeError) as e:
            print(e)
            print("Could not find or load existing NN")
            return False

    def forward(self, inputs, hidden=None):
        features = self.conv(inputs).view(inputs.size()[0], -1)
        return self.fc(features), None

    def _get_conv_out(self, shape):
        # Useful to compute the shape out of the conv blocks
        # (including eventual padding..)
        o = self.conv(torch.zeros(1, *shape))
        return int(np.prod(o.size()))
