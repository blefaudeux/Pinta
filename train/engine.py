#!/usr/local/bin/python3
"""
Implement different NNs which best describe the behaviour of the system
"""

import logging
from typing import Any, Dict, List

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from data_processing.training_set import TrainingSample, TrainingSet, TrainingSetBundle
from data_processing.transforms import Normalize
from settings import dtype


class NN(nn.Module):
    """
    This is a generic NN implementation.
    This is an abstract class, a proper model needs to be implemented
    """

    def __init__(self, logdir, log_channel="NN"):
        super(NN, self).__init__()
        self.model = None
        self._valid = False
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

    def evaluate(self, dataloader: DataLoader, settings: Dict[str, Any]):
        #  Re-use PyTorch losses on the fly
        criterion = nn.MSELoss()
        losses = []

        for seq in dataloader:
            out, _ = self(seq.inputs)
            loss = criterion(out, seq.outputs.view(out.size()[0], -1))
            losses.append(loss.item())

        return losses

    @staticmethod
    def prepare_data(training_sets: List[TrainingSet], seq_len):
        """
        Prepare sequences of a given length given the input data
        """

        bundle = TrainingSetBundle(training_sets)
        mean, std = bundle.get_norm()

        return bundle.get_sequences(seq_len), mean, std

    def fit(
        self,
        trainer: DataLoader,
        tester: DataLoader,
        settings: Dict[str, Any],
        epochs=50,
    ):
        optimizer = optim.SGD(self.parameters(), lr=0.1)
        criterion = nn.MSELoss()

        self.log.info("Training the network...\n")
        i_log = 0
        for epoch in range(epochs):

            self.log.info("***** Epoch %d", epoch)

            for train_batch, test_batch in zip(trainer, tester):
                # Eval computation on the training data
                def closure_train(data=train_batch):
                    optimizer.zero_grad()
                    out, _ = self(data.inputs)
                    loss = criterion(out.squeeze(), data.outputs)

                    self.log.info("  Train loss: {:.4f}".format(loss.item()))
                    self.summary_writer.add_scalar("train", loss.item(), i_log)
                    # Add to the gradient
                    loss.backward()
                    return loss

                # Loss on the test data
                def closure_test(data=test_batch):
                    pred, _ = self(data.inputs)
                    loss = criterion(pred.squeeze(), data.outputs.squeeze())
                    self.summary_writer.add_scalar("test", loss.item(), i_log)
                    self.log.info("  Test loss: {:.4f}\n".format(loss.item()))

                optimizer.step(closure_train)
                closure_test()

                i_log += 1

            # Update learning rate if needed
            if not (epoch + 1) % settings["training"]["lr_period_decrease"]:
                self.log.info("  -- Reducing learning rate")
                for group in optimizer.param_groups:
                    group["lr"] *= settings["training"]["lr_amount_decrease"]

            # Display the layer weights
            weight = self.get_layer_weights()
            if weight is not None:
                self.summary_writer.add_histogram("weights", weight, i_log)

        self.log.info("... Done")

    def predict(
        self,
        dataloader: DataLoader,
        mean: torch.Tensor = None,
        std: torch.Tensor = None,
    ):
        predictions = torch.stack([self(test_seq.inputs)[0] for test_seq in dataloader])

        # De-normalize the output
        if mean and std:
            return torch.add(torch.mul(predictions, std), mean)

        return predictions

    def forward(self, inputs, *kwargs):
        """
        Defines the computation performed at every call.
        Should be overriden by all subclasses.
        """
        raise NotImplementedError
