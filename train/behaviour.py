#!/usr/local/bin/python3
"""
Implement different NNs which best describe the behaviour of the system
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


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

    def fit(self, train, test, epoch=50, batch_size=1, verbose=2):
        optimizer = optim.Adam(self.parameters())
        criterion = nn.MSELoss()

        print("Training the network...")

        for i in range(epoch):
            print('epoch {}'.format(i))

            # Eval computation on the training data
            def closure():
                optimizer.zero_grad()
                out = self(train[0])
                loss = criterion(out, train[1])
                print('Eval loss: {}'.format(loss.data.numpy()[0]))
                loss.backward()
                return loss

            optimizer.step(closure)

            # Loss on the test data
            pred = self(test[0])
            loss = criterion(pred, test[1])
            print("Test loss: {}".format(loss.data.numpy()[0]))

        print("... Done")

    def predict(self, inputs):
        return self(inputs)

    def forward(self, *inputs):
        """Defines the computation performed at every call.

        Should be overriden by all subclasses.

        .. note::
            Although the recipe for forward pass needs to be defined within
            this function, one should call the :class:`Module` instance afterwards
            instead of this since the former takes care of running the
            registered hooks while the latter silently ignores them.
        """
        raise NotImplementedError


# -----------------------
#  Conv1D
class ConvNN(NN):
    def __init__(self, filename=None, input_size=3, conv_window=10,
                 n_layers=3):
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
        self.hidden_size = input_size * 4
        self.output_size = 1
        self.n_layers = n_layers

        # First conv1d + pooling
        self.conv1 = nn.Conv1d(input_size, self.hidden_size, conv_window)
        self.pool1 = nn.AvgPool1d(2)

        # Second conv1d + pooling - the time scale is effectively lengthened
        self.conv2 = nn.Conv1d(input_size * 2, input_size * 2, conv_window)
        self.pool2 = nn.AvgPool1d(2)

        self.out = nn.Linear(input_size, 1)

    def forward(self, inputs, hidden=None):
        # Run through Conv1d and Pool layers
        conv_results = self.conv1(inputs)
        pool_results = self.pool1(conv_results)
        conv_results = self.conv2(pool_results)
        pool_results = self.pool2(conv_results)

        pool_results = F.tanh(pool_results)
        output, hidden = self.gru(pool_results, hidden)
        return F.tanh(self.out(output)), hidden
