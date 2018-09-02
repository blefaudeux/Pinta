#!/usr/local/bin/python3
"""
Implement different NNs which best describe the behaviour of the system
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np


# dtype = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor
dtype = torch.FloatTensor


class NN(nn.Module):

    """
    This is a generic NN implementation.
    This is an abstract class, a proper model needs to be implemented
    """

    def __init__(self):
        super(NN, self).__init__()
        self.model = None
        self.valid = False

    def load(self, filename):
        try:
            with open(filename, "r") as f:
                self.load_state_dict(torch.load(f))
                print("---\nNetwork {} loaded".format(filename))
                print(self.model.summary())
                return True

        except (ValueError, OSError, IOError, TypeError) as _:
            print("Could not find or load existing NN")
            return False

    def save(self, name):
        with open(name, "w") as f:
            torch.save(self.state_dict(), f)

    def evaluate(self, data, batch_size=50):
        batched = self.prepare_data(data, batch_size)
        criterion = nn.MSELoss()
        out, _ = self(batched[0])
        loss = criterion(out, batched[1])
        return loss.data[0]

    @staticmethod
    def prepare_data(train, batch_size):
        """
        Prepare mini-batches out of the data sequences

        input: [2], first element is the inputs, second the outputs
        """

        n_batch = train[0].shape[1] // batch_size
        n_samples = n_batch * batch_size
        batch_data = [np.array(np.split(train[0][:, :n_samples], batch_size, axis=1)),
                      np.array(np.split(train[1][:, :n_samples], batch_size, axis=1))]

        return [Variable(torch.from_numpy(np.swapaxes(batch_data[0], 1, 2))).type(dtype),
                Variable(torch.from_numpy(np.swapaxes(batch_data[1], 1, 2))).type(dtype)]

    def fit(self, train, test, epoch=50, batch_size=50):
        optimizer = optim.Adam(self.parameters())
        criterion = nn.MSELoss()

        # Prepare the data in batches
        print("Preparing dataset...")
        train_batch = self.prepare_data(train, batch_size)
        test_batch = self.prepare_data(test, batch_size)

        print("Training the network...")

        for i in range(epoch):
            print('epoch {}'.format(i))

            # Eval computation on the training data
            def closure():
                optimizer.zero_grad()
                out, _ = self(train_batch[0])
                loss = criterion(out, train_batch[1])
                print('Eval loss: {:.4f}'.format(loss.data[0]))
                loss.backward()
                return loss

            optimizer.step(closure)

            # Loss on the test data
            pred, _ = self(test_batch[0])
            loss = criterion(pred, test_batch[1])
            print("Test loss: {:.4f}\n".format(loss.data[0]))

        print("... Done")

    def predict(self, data, batch_size=50):
        return self(self.prepare_data(data, batch_size)[0])

    def forward(self, *inputs):
        """
        Defines the computation performed at every call.
        Should be overriden by all subclasses.
        """
        raise NotImplementedError

    @staticmethod
    def _conv_out(in_dim, kernel_size, stride=1, padding=0, dilation=1):
        return np.floor((in_dim + 2 * padding - dilation *
                         (kernel_size - 1) - 1) / stride + 1)


class ConvRNN(NN):
    """
    Combination of a convolutional front end and an RNN (GRU) layer below
     >> see https://gist.github.com/spro/c87cc706625b8a54e604fb1024106556

    """

    def __init__(self, input_size, hidden_size, filename=None, n_layers=1):
        super(ConvRNN, self).__init__()

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
        # TODO: Add batch normalization ?

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = 1
        self.gru_layers = n_layers

        # Conv front end
        # First conv is a depthwise convolution
        self.conv1 = nn.Conv1d(input_size, hidden_size,
                               kernel_size=20, padding=7, groups=input_size)

        self.conv2 = nn.Conv1d(hidden_size, hidden_size,
                               kernel_size=10, padding=7)

        self.conv3 = nn.Conv1d(hidden_size, hidden_size,
                               kernel_size=2, stride=2, padding=7)

        self.relu = nn.ReLU()

        # GRU / LSTM layers
        self.gru = nn.GRU(hidden_size, hidden_size, n_layers, dropout=0.01)

        # Ends with a fully connected layer
        self.out = nn.Linear(hidden_size, self.output_size)

        # CUDA switch > Needs to be done after the model has been declared
        if dtype == torch.cuda.FloatTensor:
            print("Using Pytorch CUDA backend")
            self.cuda()

    def forward(self, inputs, hidden=None):
        batch_size = inputs.size(1)

        # Turn (seq_len x batch_size x input_size)
        # into (batch_size x input_size x seq_len) for CNN
        inputs = inputs.transpose(0, 1).transpose(1, 2)

        # Run through Conv1d and Pool1d layers
        c1 = self.conv1(inputs)
        r1 = self.relu(c1)

        c2 = self.conv2(r1)
        r2 = self.relu(c2)

        # c3 = self.conv3(r2)
        # r3 = self.relu(c3)

        # Turn (batch_size x hidden_size x seq_len) back into (seq_len x batch_size x hidden_size)
        # for GRU/LSTM layer
        r2 = r2.transpose(1, 2).transpose(0, 1)

        p = F.tanh(r2)
        output, hidden = self.gru(p, hidden)
        conv_seq_len = output.size(0)

        # Treating (conv_seq_len x batch_size) as batch_size for linear layer
        output = output.view(conv_seq_len * batch_size, self.hidden_size)
        output = torch.tanh(self.out(output))
        output = output.view(conv_seq_len, -1, self.output_size)
        return output, hidden
