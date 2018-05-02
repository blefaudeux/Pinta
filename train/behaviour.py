#!/usr/local/bin/python3
"""
Implement different NNs which best describe the behaviour of the system
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
import numpy as np


class NN(nn.Module):

    """
    This is a generic NN implementation.
    This is an abstract class, a proper model needs to be implemented
    """

    def __init__(self):
        super(NN, self).__init__()
        self._model = None  # We keep the actual NN private, safer
        self.valid = False

    def load(self, filename):
        try:
            self.load_state_dict(torch.load(filename))
            print("---\nNetwork {} loaded".format(filename))
            print(self._model.summary())
            return True

        except (ValueError, OSError, IOError) as _:
            print("Could not find or load existing NN")
            return False

    def save(self, name):
        torch.save(self._model.state_dict(), name)

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

        return [Variable(torch.from_numpy(np.swapaxes(np.swapaxes(batch_data[0], 0, 1), 0, 2)))
                .float(),
                Variable(torch.from_numpy(np.swapaxes(np.swapaxes(batch_data[1], 0, 1), 0, 2)))
                .float()]

    def fit(self, train, test, epoch=50, batch_size=100):
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
                print('Eval loss: {}'.format(loss.data.numpy()[0]))
                loss.backward()
                return loss

            optimizer.step(closure)

            # Loss on the test data
            pred = self(test_batch[0])
            loss = criterion(pred, test_batch[1])
            print("Test loss: {}".format(loss.data.numpy()[0]))

        print("... Done")

    def predict(self, inputs):
        return self(inputs)

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
            else:
                print(
                    "Could not load the specified net, \
                    computing it from scratch"
                )

        # ----
        # Define the model
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = 1
        self.gru_layers = n_layers

        self.c1 = nn.Conv1d(input_size, hidden_size, 2)
        self.p1 = nn.AvgPool1d(2)
        self.c2 = nn.Conv1d(hidden_size, hidden_size, 1)
        self.p2 = nn.AvgPool1d(2)
        self.gru = nn.GRU(hidden_size, hidden_size, n_layers, dropout=0.01)

        # Fully connected layer
        self.out = nn.Linear(hidden_size, self.output_size)

    def forward(self, inputs, hidden=None):
        batch_size = inputs.size(1)

        # Turn (seq_len x batch_size x input_size) into (batch_size x input_size x seq_len) for CNN
        inputs = inputs.transpose(0, 1).transpose(1, 2)

        # Run through Conv1d and Pool1d layers
        c = self.c1(inputs)
        p = self.p1(c)
        c = self.c2(p)
        p = self.p2(c)

        # Turn (batch_size x hidden_size x seq_len) back into (seq_len x batch_size x hidden_size)
        p = p.transpose(1, 2).transpose(0, 1)

        p = F.tanh(p)
        output, hidden = self.gru(p, hidden)
        conv_seq_len = output.size(0)

        # Treating (conv_seq_len x batch_size) as batch_size for linear layer
        output = output.view(conv_seq_len * batch_size, self.hidden_size)
        output = F.tanh(self.out(output))
        output = output.view(conv_seq_len, -1, self.output_size)
        return output, hidden


class ConvNN(NN):
    """
    [Broken]
    Purely conv nn, the stride is used in place of the pooling layers

    Arguments:
        NN {[type]} -- [description]
    """

    def __init__(self, filename=None, input_size=3, conv_window=10):
        super(ConvNN, self).__init__()

        # Load from trained NN if required
        if filename is not None:
            self.valid = self.load(filename)
            if self.valid:
                return
            else:
                print(
                    "Could not load the specified net, \
                    computing it from scratch"
                )

        # ----
        # Define the model
        self.input_size = input_size
        self.hidden_size = input_size * conv_window
        self.output_size = 1

        self.relu = nn.ReLU()

        self.conv1 = nn.Conv1d(input_size, self.hidden_size,
                               conv_window, groups=input_size)

        self.conv2 = nn.Conv1d(self.hidden_size, self.hidden_size * conv_window,
                               conv_window, groups=1, stride=2)

        self.conv3 = nn.Conv1d(self.hidden_size, self.hidden_size * conv_window,
                               conv_window, groups=1, stride=2)

        # Fully connected layer
        self.out = nn.Linear(input_size, 1)

    def forward(self, inputs, hidden=None):
        variable_input = Variable(
            torch.from_numpy(inputs).float(), requires_grad=False)

        # Run through Conv1d layers, relu the outputs
        conv_results = self.relu(self.conv1(variable_input))
        conv2_results = self.relu(self.conv2(conv_results))
        return self.out(F.tanh(conv2_results)), hidden


class LSTM(NN):

    def __init__(self):
        super(LSTM, self).__init__()
        self.lstm1 = nn.LSTMCell(3, 51)
        self.lstm2 = nn.LSTMCell(51, 51)
        self.linear = nn.Linear(51, 1)

    def forward(self, input, future=0):
        outputs = []
        h_t = Variable(
            torch.zeros(input.size(0), 51).double(), requires_grad=False)

        c_t = Variable(
            torch.zeros(input.size(0), 51).double(), requires_grad=False)

        h_t2 = Variable(
            torch.zeros(input.size(0), 51).double(), requires_grad=False)

        c_t2 = Variable(
            torch.zeros(input.size(0), 51).double(), requires_grad=False)

        for _, input_t in enumerate(input.chunk(input.size(1), dim=1)):
            h_t, c_t = self.lstm1(input_t, (h_t, c_t))
            h_t2, c_t2 = self.lstm2(h_t, (h_t2, c_t2))
            output = self.linear(h_t2)
            outputs += [output]

        for _ in range(future):  # if we should predict the future
            h_t, c_t = self.lstm1(output, (h_t, c_t))
            h_t2, c_t2 = self.lstm2(h_t, (h_t2, c_t2))
            output = self.linear(h_t2)
            outputs += [output]

        outputs = torch.stack(outputs, 1).squeeze(2)
        return outputs
