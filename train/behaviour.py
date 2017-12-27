#!/usr/local/bin/python3
"""
Implement different NNs which best describe the behaviour of the system
"""

import torch
import torch.nn as nn


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
            self._model.load_state_dict(torch.load(filename))
            print("---\nNetwork {} loaded".format(filename))
            print(self._model.summary())
            return True

        except (ValueError, OSError, IOError) as _:
            print("Could not find or load existing NN")
            return False

    def save(self, name):
        torch.save(self._model.state_dict(), name)

    def fit(self, train_in, train_out, epoch=50, batch_size=1, verbose=2):

        print("Training the network...")
        self._model.fit(
            train_in,
            train_out,
            epoch=epoch,
            batch_size=batch_size,
            verbose=verbose
        )
        self.valid = True
        print("... Done")

    def predict(self, inputs):
        return self._model.predict(inputs)

    def evaluate(self, inputs, outputs, verbose):
        # More settings could be exposed here. Needed ?
        return self._model.evaluate(inputs, outputs, verbose=verbose)

    def forward(self, inputs, hidden):
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
#  Conv1D implementation
class ConvNN(NN):
    def __init__(self, filename=None, input_size=3, conv_window=10, n_layers=3):
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

        # Else define the _model
        self.input_size = input_size
        self.hidden_size = conv_window
        self.output_size = 1
        self.n_layers = n_layers

        self.conv1 = nn.Conv1d(input_size, conv_window, 2)
        self.pool1 = nn.AvgPool1d(2)
        self.conv2 = nn.Conv1d(conv_window, conv_window, 1)
        self.pool2 = nn.AvgPool1d(2)
        self.gru = nn.GRU(conv_window, conv_window, n_layers, dropout=0.01)
        self.out = nn.Linear(conv_window, 1)


    def forward(self, inputs, hidden):
        batch_size = inputs.size(1)
        
        # Turn (seq_len x batch_size x input_size) into (batch_size x input_size x seq_len) for CNN
        inputs = inputs.transpose(0, 1).transpose(1, 2)

        # Run through Conv1d and Pool1d layers
        c = self.c1(inputs)
        p = self.p1(c)
        c = self.c2(p)
        p = self.p2(c)

        # Turn (batch_size x hidden_size x seq_len) back into (seq_len x batch_size x hidden_size) for RNN
        p = p.transpose(1, 2).transpose(0, 1)
        
        p = F.tanh(p)
        output, hidden = self.gru(p, hidden)
        conv_seq_len = output.size(0)
        output = output.view(conv_seq_len * batch_size, self.hidden_size) # Treating (conv_seq_len x batch_size) as batch_size for linear layer
        output = F.tanh(self.out(output))
        output = output.view(conv_seq_len, -1, self.output_size)
        return output, hidden


# -----------------------
# LSTM Implementation
class MemoryNN(NN):
    def __init__(self, filename=None):
        super(MemoryNN, self).__init__()

        # Load from trained NN if required
        if filename is not None:
            self.valid = self.load(filename)

        if self.valid:
            return

        # Else define the _model
        hidden_neurons = 32
        self._model = Sequential()
        self._model.add(Dense(hidden_neurons, input_dim=3, activation='relu'))
        self._model.add(LSTM(hidden_neurons, return_sequences=False))
        self._model.add(Dense(hidden_neurons, activation='relu'))
        self._model.add(Dense(hidden_neurons, activation='relu'))
        self._model.add(Dense(hidden_neurons, activation='relu'))
        self._model.add(Dense(1, input_dim=hidden_neurons, activation='linear'))

        self._model.compile(loss="mean_squared_error", optimizer="rmsprop")
