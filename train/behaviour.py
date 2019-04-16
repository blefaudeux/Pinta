#!/usr/local/bin/python3
"""
Implement different NNs which best describe the behaviour of the system
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
import numpy as np
import logging
from tensorboardX import SummaryWriter

dtype = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor

if dtype == torch.cuda.FloatTensor:
    print("CUDA enabled")
else:
    print("CPU enabled")


class NN(nn.Module):

    """
    This is a generic NN implementation.
    This is an abstract class, a proper model needs to be implemented
    """

    def __init__(self, logdir):
        super(NN, self).__init__()
        self.model = None
        self.valid = False
        self.mean = None
        self.std = None

        # Set up TensorBoard
        self.summary_writer = SummaryWriter(logdir)
        self.summary_writer.add_graph(self.model)

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
        with open(name, "wb") as f:
            torch.save(self.state_dict(), f)

    def evaluate(self, data, batch_size=50):
        batched, _, _ = self.prepare_data(data, batch_size, normalize=False)
        criterion = nn.MSELoss()
        out, _ = self(batched[0])
        loss = criterion(out, batched[1])
        return loss.item()

    @staticmethod
    def prepare_data(train, batch_size, normalize=True):
        """
        Prepare mini-batches out of the data sequences

        train: [2], first element is the inputs, second the outputs
        """

        # Compute some stats on the data
        mean = [np.mean(train[0], axis=1), np.mean(train[1], axis=1)]
        std = [np.std(train[0], axis=1), np.std(train[1], axis=1)]

        if normalize:
            # Normalize the data, bring it back to zero mean and STD of 1
            train[0] = np.subtract(train[0].transpose(), mean[0]).transpose()
            train[1] = np.subtract(train[1].transpose(), mean[1]).transpose()

            train[0] = np.divide(train[0].transpose(), std[0]).transpose()
            train[1] = np.divide(train[1].transpose(), std[1]).transpose()
            print("Data normalized")

        # Compute normalized batches
        n_batch = train[0].shape[1] // batch_size
        n_samples = n_batch * batch_size
        batch_data = [np.array(np.split(train[0][:, :n_samples], batch_size, axis=1)),
                      np.array(np.split(train[1][:, :n_samples], batch_size, axis=1))]

        # Return [batch data], mean, std
        mean = [torch.from_numpy(mean[0]).type(dtype),
                torch.from_numpy(mean[1]).type(dtype)]

        std = [torch.from_numpy(std[0]).type(dtype),
               torch.from_numpy(std[1]).type(dtype)]

        return [Variable(torch.from_numpy(np.swapaxes(batch_data[0], 1, 2))).type(dtype),
                Variable(torch.from_numpy(np.swapaxes(batch_data[1], 1, 2))).type(dtype)], mean, std

    def fit(self, train, test, epoch=50, batch_size=50):
        optimizer = optim.Adam(self.parameters())
        criterion = nn.MSELoss()

        # Prepare the data in batches
        print("Preparing dataset...")
        train_batch, self.mean, self.std = self.prepare_data(train,
                                                             batch_size,
                                                             normalize=True)

        test_batch, _, _ = self.prepare_data(test,
                                             batch_size,
                                             normalize=False)

        # Test data needs to be normalized with the same coefficients as training data
        test_batch[0] = torch.div(torch.add(test_batch[0], - self.mean[0]),
                                  self.std[0])

        test_batch[1] = torch.div(torch.add(test_batch[1], - self.mean[1]),
                                  self.std[1])
        print("Training the network...")

        for i in range(epoch):
            print('epoch {}'.format(i))

            # Eval computation on the training data
            def closure():
                optimizer.zero_grad()
                out, _ = self(train_batch[0])
                loss = criterion(out, train_batch[1])
                print('Train loss: {:.4f}'.format(loss.item()))
                self.summary_writer.add_scalar('train', loss.item(), i)
                loss.backward()
                return loss

            optimizer.step(closure)

            # Loss on the test data
            pred, _ = self(test_batch[0])
            loss = criterion(pred, test_batch[1])
            self.summary_writer.add_scalar('test', loss.item(), i)
            print("Test loss: {:.4f}\n".format(loss.item()))

        print("... Done")

    def predict(self, data, batch_size=50):
        # batch and normalize
        batched, _, _ = self.prepare_data(data, batch_size, normalize=False)

        batched = torch.div(torch.add(batched[0], - self.mean[0]),
                            self.std[0])

        # De-normalize the output
        return torch.add(
            torch.mul(self(batched)[0], self.std[1]),
            self.mean[1]
        ).detach().cpu().numpy()

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

    def __init__(self, logdir, input_size, hidden_size, filename=None, n_gru_layers=1):
        super(ConvRNN, self).__init__(logdir)

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
        self.gru_layers = n_gru_layers

        # Conv front end
        # First conv is a depthwise convolution
        self.conv1 = nn.Conv1d(input_size, hidden_size,
                               kernel_size=10, padding=3, groups=input_size)

        self.conv2 = nn.Conv1d(hidden_size, hidden_size,
                               kernel_size=6, padding=4)

        self.relu = nn.ReLU()

        # GRU / LSTM layers
        self.gru = nn.GRU(hidden_size, hidden_size, n_gru_layers, dropout=0.01)

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

        # Turn (batch_size x hidden_size x seq_len) back into (seq_len x batch_size x hidden_size)
        # for GRU/LSTM layer
        r2 = r2.transpose(1, 2).transpose(0, 1)

        output, hidden = self.gru(r2, hidden)
        conv_seq_len = output.size(0)

        # Treating (conv_seq_len x batch_size) as batch_size for linear layer
        output = output.view(conv_seq_len * batch_size, self.hidden_size)
        output = torch.tanh(self.out(output))
        output = output.view(conv_seq_len, -1, self.output_size)
        return output, hidden


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
                               kernel_size=10, padding=3, groups=input_size)

        self.conv2 = nn.Conv1d(hidden_size, hidden_size,
                               kernel_size=6, padding=4)

        self.relu = nn.ReLU()

        # Ends with a fully connected layer
        self.out = nn.Linear(hidden_size, self.output_size)

        # CUDA switch > Needs to be done after the model has been declared
        if dtype == torch.cuda.FloatTensor:
            print("Using Pytorch CUDA backend")
            self.cuda()

    def forward(self, inputs, hidden=None):
        batch_size = inputs.size(0)

        # Turn (seq_len x batch_size x input_size)
        # into (batch_size x input_size x seq_len) for CNN
        inputs = inputs.transpose(0, 1).transpose(1, 2)

        # Run through Conv1d and Pool1d layers
        c1 = self.conv1(inputs)
        r1 = self.relu(c1)

        c2 = self.conv2(r1)
        r2 = self.relu(c2)

        # Treating (conv_seq_len x batch_size) as batch_size for linear layer
        output = r2.view(r2.size(0) * batch_size, self.hidden_size)
        output = torch.tanh(self.out(output))
        output = output.view(batch_size, -1, self.output_size)
        return output, hidden
