import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
import numpy as np
import logging
from tensorboardX import SummaryWriter

from train.behaviour import NN

dtype = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor


class Conv(NN):
    """
    Pure Conv
    """

    def __init__(self, logdir, input_size, hidden_size, filename=None):
        super(Conv, self).__init__(logdir)

        # Load from trained NN if required
        if filename is not None:
            self.valid = self.load(filename)
            if self.valid:
                return

            print(
                "Could not load the specified net, computing it from scratch"
            )

        # ----
        # Define the model
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = 1
        KERNEL_SIZE = 2

        # Conv front end
        # First conv is a depthwise convolution
        self.conv = nn.Sequential(nn.Conv1d(input_size[1], hidden_size,
                                            kernel_size=KERNEL_SIZE),
                                  nn.ReLU(),
                                  nn.Conv1d(hidden_size, hidden_size,
                                            kernel_size=KERNEL_SIZE),
                                  nn.ReLU())

        out_conv_size = self._get_conv_out(input_size)

        # Ends with a fully connected layer
        self.fc = nn.Sequential(nn.Linear(out_conv_size, 512),
                                nn.ReLU(),
                                nn.Linear(out_conv_size, self.output_size))

        # CUDA switch > Needs to be done after the model has been declared
        if dtype is torch.cuda.FloatTensor:
            print("Using Pytorch CUDA backend")
            self.cuda()

    def _get_conv_out(self, shape):
        # Useful to compute the shape out of the conv blocks (including eventual padding..)
        # on the fly
        o = self.conv(torch.zeros(*shape))
        return int(np.prod(o.size()))

    def forward(self, inputs, hidden=None):
        features = self.conv(inputs).view(inputs.size()[0], -1)
        return self.fc(features), None
