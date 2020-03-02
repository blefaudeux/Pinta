
import torch
import torch.nn as nn

from settings import dtype
from train.engine import NN


class ConvRNN(NN):
    """
    Combination of a convolutional front end and an RNN (GRU) layer below
     >> see https://gist.github.com/spro/c87cc706625b8a54e604fb1024106556

    """

    def __init__(self,
                 logdir,
                 input_size,
                 hidden_size,
                 filename=None,
                 n_gru_layers=1):
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
            print("Using Pytorch CUDA backend."
                  "Moving the net definition to device")
            self.cuda()

    def forward(self, inputs, hidden=None):
        batch_size = inputs.size(0)

        # Turn (batch_size x batch_number x input_size)
        # into (batch_size x input_size x batch_number) for CNN
        inputs = inputs.transpose(1, 2)

        # Run through Conv1d and Pool1d layers
        r1 = self.relu(self.conv1(inputs))
        r2 = self.relu(self.conv2(r1))

        # Turn  (batch_size x input_size x batch_number)
        # back into (batch_size x batch_number x input_size)
        # for GRU/LSTM layer
        r2 = r2.transpose(1, 2)

        output, hidden = self.gru(r2, hidden)
        conv_seq_len = output.size(2)

        # Treating (conv_seq_len x batch_size) as batch_size for linear layer
        output = output.view(
            batch_size//self.hidden_size, -1, self.hidden_size)
        output = self.out(output)
        output = output.view(conv_seq_len, -1, self.output_size)
        return output, hidden

    def get_layer_weights(self):
        return self.conv1.weight
