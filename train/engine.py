#!/usr/local/bin/python3
"""
Implement different NNs which best describe the behaviour of the system
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from tensorboardX import SummaryWriter
from settings import dtype
from data_processing.training_set import TrainingSet, TrainingSetBundle, TrainingSample
from typing import List


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

    @property
    def valid(self):
        return self._valid

    def save(self, name):
        with open(name, "wb") as f:
            torch.save(self.state_dict(), f)

    def updateNormalization(self, settings):
        assert "dataset_normalization" in settings.keys()

        # Update reference mean and std
        self.mean = TrainingSample(
            torch.Tensor(settings["dataset_normalization"]["input"]["mean"]).type(dtype),
            torch.Tensor(settings["dataset_normalization"]["output"]["mean"]).type(dtype))

        self.std = TrainingSample(
            torch.Tensor(settings["dataset_normalization"]["input"]["std"]).type(dtype),
            torch.Tensor(settings["dataset_normalization"]["output"]["std"]).type(dtype))

    def evaluate(self, data, settings):
        # Move the data to the proper format
        data_seq, _, _ = self.prepare_data(data, settings["seq_length"],
                                           self_normalize=False)

        data_seq.normalize(self.mean, self.std)

        # Re-use PyTorch losses on the fly
        criterion = nn.MSELoss()
        out, _ = self(data_seq.inputs)
        loss = criterion(out, data_seq.outputs.view(out.size()[0], -1))
        return loss.item()

    @staticmethod
    def prepare_data(training_sets: List[TrainingSet], seq_len, self_normalize=True):
        """
        Prepare sequences of a given length given the input data
        """

        bundle = TrainingSetBundle(training_sets)
        mean, std = bundle.get_norm()

        # except IndexError:
        #     # The data is not packed in a tensor, we need to generate one on the fly
        #     mean = [data_list.input, data_list.output]
        #     std = [np.array([1.]), np.array([1.])]
        #     pack_in = np.array([[data_list.input for i in range(seq_len)]])
        #     pack_out = np.array([[data_list.output for i in range(seq_len)]])
        #     data_list = TrainingSample(pack_in, pack_out)

        if self_normalize:
            bundle.normalize()

        training_data = bundle.get_sequences(seq_len)

        return training_data, mean, std

    @staticmethod
    def randomize_data(training_set):
        # Randomize the input/output pairs
        assert training_set.input.shape[0] == training_set.output.shape[0]

        shuffle = torch.randperm(training_set.input.shape[0])
        return TrainingSet(training_set.input[shuffle], training_set.output[shuffle])

    def fit(self, train, test, settings, epoch=50, batch_size=50, self_normalize=False):
        optimizer = optim.SGD(self.parameters(), lr=0.01)
        criterion = nn.MSELoss()

        # Prepare the data in batches
        if not self_normalize:
            # Use the normalization defined in the settings
            train_seq, _, _ = self.prepare_data(train,
                                                settings["seq_length"],
                                                self_normalize=False)

            train_seq.normalize(self.mean, self.std)
        else:
            # Compute the dataset normalization on the fly
            train_seq, self.mean, self.std = self.prepare_data(train,
                                                               settings["seq_length"],
                                                               self_normalize=True)

        train_seq.randomize()

        test_seq, _, _ = self.prepare_data(test,
                                           settings["seq_length"],
                                           self_normalize=False)

        # Test data needs to be normalized with the same coefficients as training data
        test_seq.normalize(self.mean, self.std)

        print("Training the network...")
        i_log = 0
        for i in range(epoch):
            print(f'\n***** Epoch {i}')

            for batch_index in range(0, train_seq.inputs.shape[0], batch_size):
                # Eval computation on the training data
                def closure():
                    data = TrainingSet(train_seq.inputs[batch_index:batch_index+batch_size, :, :],
                                       train_seq.outputs[batch_index:batch_index+batch_size, :])

                    optimizer.zero_grad()
                    out, _ = self(data.inputs)
                    loss = criterion(out, data.outputs)
                    print('Train loss: {:.4f}'.format(loss.item()))
                    self.summary_writer.add_scalar('train', loss.item(), i_log)
                    # Add to the gradient
                    loss.backward()
                    return loss

                optimizer.step(closure)

                # Loss on the test data
                pred, _ = self(test_seq.inputs)
                loss = criterion(pred, test_seq.outputs)
                self.summary_writer.add_scalar('test', loss.item(), i_log)
                print("Test loss: {:.4f}\n".format(loss.item()))
                i_log += 1

            # Update learning rate if needed
            if not (i + 1) % settings["training"]["lr_period_decrease"]:
                print("Reducing learning rate")
                for g in optimizer.param_groups:
                    g['lr'] *= settings["training"]["lr_amount_decrease"]

        print("... Done")

    def predict(self, data, seq_len=100):
        # FIXME: Ben - this is broken and clumsy
        # if not isinstance(data.inputs, list) and data.inputs.size == data.inputs.shape[0]:
        #     # Only one sample, need some -constant- padding
        #     data = TrainingSet([np.repeat(np.array([data.inputs]), seq_len, axis=0).transpose()],
        #                        [np.repeat(np.array([data.outputs]), seq_len, axis=0).transpose()])

        # batch and normalize
        test_seq, _, _ = self.prepare_data(data, seq_len, self_normalize=False)
        test_seq.normalize(self.mean, self.std)

        # De-normalize the output
        return torch.add(
            torch.mul(self(test_seq.inputs)[0], self.std[1]),
            self.mean[1]
        )

    def forward(self, *inputs):
        """
        Defines the computation performed at every call.
        Should be overriden by all subclasses.
        """
        raise NotImplementedError
