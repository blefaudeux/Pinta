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
from settings import Dataframe

# Handle GPU compute if available
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
        self._valid = False
        self.mean = None
        self.std = None

        # Set up TensorBoard
        self.summary_writer = SummaryWriter(logdir)
        self.summary_writer.add_graph(self.model)

    def valid(self):
        return self._valid

    def save(self, name):
        with open(name, "wb") as f:
            torch.save(self.state_dict(), f)

    def normalize(self, dataframe):
        return Dataframe(
            torch.div(
                torch.add(dataframe.input, - self.mean[0].reshape(1, -1, 1)),
                self.std[0].reshape(1, -1, 1)),
            torch.div(
                torch.add(dataframe.output, - self.mean[1].reshape(1, -1, 1)),
                self.std[1].reshape(1, -1, 1)))

    def evaluate(self, data, seq_len=100):
        data_seq, _, _ = self.prepare_data(data, seq_len, normalize=False)
        data_seq = self.normalize(data_seq)

        criterion = nn.MSELoss()
        out, _ = self(data_seq.input)
        loss = criterion(out, data_seq.output.view(out.size()[0], -1))
        return loss.item()

    @staticmethod
    def prepare_data(data_list, seq_len, normalize=True):
        """
        Prepare sequences of a given length given the input data
        """

        # Compute some stats on the data
        mean = np.mean(np.array([[np.mean(t, axis=1) for t in data_list.input], [
                       np.mean(t, axis=1) for t in data_list.output]]), axis=1)

        std = np.mean(np.array([[np.std(t, axis=1) for t in data_list.input], [
            np.std(t, axis=1) for t in data_list.output]]), axis=1)

        if normalize:
            data_normalize = Dataframe([], [])

            for data in data_list.input:
                # Normalize the data, bring it back to zero mean and STD of 1
                data = np.subtract(data.transpose(), mean[0]).transpose()
                data = np.divide(data.transpose(), std[0]).transpose()
                data_normalize.input.append(data)

            for data in data_list.output:
                data = np.subtract(data.transpose(), mean[1]).transpose()
                data = np.divide(data.transpose(), std[1]).transpose()
                data_normalize.output.append(data)

            data_list = data_normalize
            print("Data normalized")

        # To Torch tensor
        mean = [torch.from_numpy(mean[0]).type(dtype),
                torch.from_numpy(mean[1]).type(dtype)]

        std = [torch.from_numpy(std[0]).type(dtype),
               torch.from_numpy(std[1]).type(dtype)]

        # Compute all the seq samples
        # TODO: Ben - take the temporal coherency constraints into account here
        n_sequences = train[0].shape[1] - seq_len
        training_data = Dataframe(torch.from_numpy(np.array([train[0][:, start:start+seq_len]
                                                             for start in range(n_sequences)])
                                                   ).type(dtype),
                                  torch.from_numpy(train[1][:, :-seq_len]).type(dtype))

        return training_data, mean, std

    def fit(self, train, test, epoch=50, batch_size=50, seq_len=100):
        optimizer = optim.SGD(self.parameters(), lr=0.01)
        criterion = nn.MSELoss()

        # Prepare the data in batches
        train_seq, self.mean, self.std = self.prepare_data(train,
                                                           seq_len,
                                                           normalize=True)

        test_seq, _, _ = self.prepare_data(test,
                                           seq_len,
                                           normalize=False)

        # Test data needs to be normalized with the same coefficients as training data
        test_seq = self.normalize(test_seq)

        print("Training the network...")

        LR_PERIOD_DECREASE = 10
        LR_AMOUNT_DECREASE = 0.05

        for i in range(epoch):
            print('\n***** Epoch {}'.format(i))

            for b in range(0, train_seq.input.shape[0], batch_size):
                # Prepare batch
                batch_data = Dataframe(train_seq.input[b:b+batch_size, :, :],
                                       train_seq.output[:, b:b+batch_size])

                # Eval computation on the training data
                def closure():
                    optimizer.zero_grad()
                    out, _ = self(batch_data.input)
                    loss = criterion(
                        out, batch_data.output.view(out.size()[0], -1))
                    print('Train loss: {:.4f}'.format(loss.item()))
                    self.summary_writer.add_scalar('train', loss.item(), i)
                    loss.backward()
                    return loss

                optimizer.step(closure)

                # Loss on the test data
                pred, _ = self(test_seq.input)
                loss = criterion(
                    pred, test_seq.output.view(pred.size()[0], -1))
                self.summary_writer.add_scalar('test', loss.item(), i)
                print("Test loss: {:.4f}\n".format(loss.item()))

            # Update learning rate if needed
            if not (i + 1) % LR_PERIOD_DECREASE:
                print("Reducing learning rate")
                for g in optimizer.param_groups:
                    g['lr'] *= LR_AMOUNT_DECREASE

        print("... Done")

    def predict(self, data, seq_len=100):
        # batch and normalize
        test_seq, _, _ = self.prepare_data(data, seq_len, normalize=False)
        test_seq = self.normalize(test_seq)

        # De-normalize the output
        return torch.add(
            torch.mul(self(test_seq.input)[0], self.std[1]),
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
