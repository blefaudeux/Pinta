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

        # Conv front end
        # First conv is a depthwise convolution
        self.conv1 = nn.Conv1d(input_size, hidden_size,
                               kernel_size=12, groups=input_size, padding=6)

        self.conv2 = nn.Conv1d(hidden_size, hidden_size,
                               kernel_size=12, padding=7)

        self.conv3 = nn.Conv1d(hidden_size, hidden_size,
                               kernel_size=5, padding=6)

        self.relu = nn.ReLU()

        # Ends with a fully connected layer
        self.out = nn.Linear(hidden_size, self.output_size)

        # CUDA switch > Needs to be done after the model has been declared
        if dtype is torch.cuda.FloatTensor:
            print("Using Pytorch CUDA backend")
            self.cuda()

    def forward(self, inputs, hidden=None):
        batch_size = inputs.size(0)

        # Run through Conv1d and Pool1d layers
        c1 = self.conv1(inputs.transpose(1, 2))
        r1 = self.relu(c1)

        c2 = self.conv2(r1)
        r2 = self.relu(c2)

        c3 = self.conv3(r2)
        r3 = self.relu(c3)

        return self.out(r3.transpose(1, 2)), hidden
