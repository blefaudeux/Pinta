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
        # TODO: Move to torch format here
        n_batch = train[0].shape[0] // batch_size
        n_samples = n_batch * batch_size
        batched_data = [np.array(np.split(train[0][:n_samples,:], batch_size)),
                        np.array(np.split(train[1][:n_samples,:], batch_size))]

        return np.moveaxis(batched_data[0], 0, 2), batched_data[1]

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
                out = self(train_batch[0])
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

# -----------------------
#  Conv1D
class ConvNN(NN):

    def __init__(self, filename=None, input_size=3, conv_window=10):
        super(ConvNN, self).__init__()

        # Load from trained NN if required
        if filename is not None:
            self.valid = self.load(filename)
            if self.valid:
                return
            else:
                print(
                    "Could not load the specified net, computing it from scratch"
                )

        # ----
        # Define the model
        self.input_size = input_size
        self.hidden_size = input_size * conv_window
        self.output_size = 1

        # First conv1d + pooling
        self.conv1 = nn.Conv1d(input_size, self.hidden_size, conv_window, groups=input_size)
        self.pool1 = nn.AvgPool1d(2)

        # Second conv1d + pooling - the time scale is effectively lengthened
        self.conv2 = nn.Conv1d(self.hidden_size, self.hidden_size, conv_window, groups=input_size)
        self.pool2 = nn.AvgPool1d(2)

        self.out = nn.Linear(input_size, 1)

    def forward(self, inputs, hidden=None):
        variable_input = Variable(
            torch.from_numpy(inputs).float(), requires_grad=False)

        # Run through Conv1d and Pool layers
        conv_results = self.conv1(variable_input)
        pool_results = self.pool1(conv_results)
        conv_results = self.conv2(pool_results)
        pool_results = self.pool2(conv_results)
        pool_results = F.tanh(pool_results)
        return self.out(F.tanh(pool_results)), hidden



class LSTM(NN):

    def __init__(self):
        super(LSTM, self).__init__()
        self.lstm1 = nn.LSTMCell(1, 51)
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

