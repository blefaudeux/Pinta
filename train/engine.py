#!/usr/local/bin/python3
"""
Implement different NNs which best describe the behaviour of the system
"""

import logging
from typing import List

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from data_processing.training_set import (TrainingSample, TrainingSet,
                                          TrainingSetBundle)
from data_processing.transforms import Normalize
from settings import dtype
from utils import timing


class NN(nn.Module):
    """
    This is a generic NN implementation.
    This is an abstract class, a proper model needs to be implemented
    """

    def __init__(self, logdir, log_channel="NN"):
        super(NN, self).__init__()
        self.model = None
        self._valid = False
        self.mean = None
        self.std = None
        self.log = logging.getLogger(log_channel)

        # Set up TensorBoard
        self.summary_writer = SummaryWriter(logdir)

    @property
    def valid(self):
        return self._valid

    def save(self, name):
        with open(name, "wb") as fileio:
            torch.save(self.state_dict(), fileio)

    def get_layer_weights(self):
        raise NotImplementedError

    def update_normalization(self, settings):
        assert "dataset_normalization" in settings.keys()

        # Update reference mean and std
        self.mean = TrainingSample(
            torch.Tensor(settings["dataset_normalization"]
                         ["input"]["mean"]).type(dtype),
            torch.Tensor(settings["dataset_normalization"]["output"]["mean"]).type(dtype))

        self.std = TrainingSample(
            torch.Tensor(settings["dataset_normalization"]
                         ["input"]["std"]).type(dtype),
            torch.Tensor(settings["dataset_normalization"]["output"]["std"]).type(dtype))

    def evaluate(self, data, settings):
        # Move the data to the proper format
        data_seq, _, _ = self.prepare_data(data, settings["seq_length"],
                                           self_normalize=False)

        data_seq.set_transforms([Normalize(self.mean, self.std)])

        # Re-use PyTorch losses on the fly
        criterion = nn.MSELoss()
        out, _ = self(data_seq.inputs)
        loss = criterion(out, data_seq.outputs.view(out.size()[0], -1))
        return loss.item()

    @staticmethod
    def prepare_data(training_sets: List[TrainingSet], seq_len):
        """
        Prepare sequences of a given length given the input data
        """

        bundle = TrainingSetBundle(training_sets)
        mean, std = bundle.get_norm()

        return bundle.get_sequences(seq_len), mean, std

    def fit(self, train, test, settings, epochs=50, batch_size=50):
        optimizer = optim.SGD(self.parameters(), lr=0.01)
        criterion = nn.MSELoss()

        # Prepare the data in batches
        # Use the normalization defined in the settings
        train_seq, mean, std = self.prepare_data(train, settings["seq_length"])
        train_seq.set_transforms([Normalize(mean, std)])

        # Test data needs to be normalized with the same coefficients as training data
        test_seq, _, _ = self.prepare_data(test, settings["seq_length"])
        test_seq.set_transforms([Normalize(mean, std)])

        train_dataload = DataLoader(
            train_seq, batch_size=batch_size, shuffle=True)

        test_dataload = DataLoader(
            test_seq, batch_size=batch_size, shuffle=True)

        self.log.info("Training the network...\n")
        i_log = 0
        for epoch in range(epochs):

            self.log.info("***** Epoch %d", epoch)

            for train_batch, test_batch in zip(train_dataload, test_dataload):
                # Eval computation on the training data
                @timing
                def closure_train(data=train_batch):
                    optimizer.zero_grad()
                    out, _ = self(data.inputs)
                    loss = criterion(out, data.outputs)

                    self.log.info('  Train loss: {:.4f}'.format(loss.item()))
                    self.summary_writer.add_scalar('train', loss.item(), i_log)
                    # Add to the gradient
                    loss.backward()
                    return loss

                # Loss on the test data
                @timing
                def closure_test(data=test_batch):
                    pred, _ = self(data.inputs)
                    loss = criterion(pred, data.outputs)
                    self.summary_writer.add_scalar('test', loss.item(), i_log)
                    self.log.info("  Test loss: {:.4f}\n".format(loss.item()))

                optimizer.step(closure_train)
                closure_test()

                i_log += 1

            # Update learning rate if needed
            if not (epoch + 1) % settings["training"]["lr_period_decrease"]:
                self.log.info("  -- Reducing learning rate")
                for group in optimizer.param_groups:
                    group['lr'] *= settings["training"]["lr_amount_decrease"]

            # Display the layer weights
            weight = self.get_layer_weights()
            if weight is not None:
                self.summary_writer.add_histogram("weights", weight, i_log)

        self.log.info("... Done")

    def predict(self, data, seq_len=100):
        if isinstance(data, TrainingSample) and data.inputs.size == data.inputs.shape[0]:
            # Only one sample, need some -constant- padding
            data = [TrainingSet.from_training_sample(data, seq_len)]

        # batch and normalize
        test_seq, _, _ = self.prepare_data(data, seq_len, self_normalize=False)
        test_seq.set_transforms([Normalize(self.mean, self.std)])

        # De-normalize the output
        return torch.add(
            torch.mul(self(test_seq.inputs)[0], self.std[1]),
            self.mean[1]
        )

    def forward(self, inputs, *kwargs):
        """
        Defines the computation performed at every call.
        Should be overriden by all subclasses.
        """
        raise NotImplementedError
