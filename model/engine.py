#!/usr/local/bin/python3
"""
Implement different NNs which best describe the behaviour of the system
"""

import logging
import time
from itertools import cycle
from typing import Any, Dict

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter


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

    def predict(
        self,
        dataloader: DataLoader,
        mean: torch.Tensor = None,
        std: torch.Tensor = None,
    ):
        predictions = [self(seq.inputs)[0] for seq in dataloader]
        predictions_tensor = torch.cat(predictions).squeeze()

        # De-normalize the output
        if mean and std:
            return torch.add(torch.mul(predictions_tensor, std), mean)

        return predictions_tensor

    def fit(
        self,
        trainer: DataLoader,
        tester: DataLoader,
        settings: Dict[str, Any],
        epochs=50,
    ):
        optimizer = optim.Adam(
            self.parameters(), lr=settings["learning_rate"], amsgrad=True
        )
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer)
        criterion = nn.MSELoss()

        if len(tester) < len(trainer):
            tester = cycle(tester)

        self.log.info("Training the network...\n")
        i_log = 0
        for i_epoch in range(epochs):

            self.log.info("***** Epoch %d", i_epoch)

            validation_loss = torch.zeros(1)

            for i_batch, (train_batch, validation_batch) in enumerate(
                zip(trainer, tester)
            ):
                batch_start = time.time()

                # Eval computation on the training data
                def closure_train(data=train_batch):
                    optimizer.zero_grad()
                    out, _ = self(data.inputs)
                    loss = criterion(out.squeeze(), data.outputs.squeeze())

                    self.log.info(
                        " {}/{},{} Train loss: {:.4f}".format(
                            i_epoch, epochs, i_batch, loss.item()
                        )
                    )
                    self.summary_writer.add_scalar("train", loss.item(), i_log)
                    # Add to the gradient
                    loss.backward()
                    return loss

                # Loss on the validation data
                def closure_validation(data=validation_batch):
                    pred, _ = self(data.inputs)
                    loss = criterion(pred.squeeze(), data.outputs.squeeze())
                    self.summary_writer.add_scalar("validation", loss.item(), i_log)
                    self.log.info(
                        " {}/{},{} Validation loss: {:.4f}".format(
                            i_epoch, epochs, i_batch, loss.item()
                        )
                    )
                    return loss

                optimizer.step(closure_train)
                validation_loss += closure_validation()

                # Time some operations
                batch_stop = time.time()
                elapsed = batch_stop - batch_start

                samples_per_sec = (
                    train_batch.inputs.shape[0] + validation_batch.inputs.shape[0]
                ) / elapsed

                self.summary_writer.add_scalar(
                    f"Samples_per_sec", samples_per_sec, i_log
                )
                self.log.info(
                    " {}/{},{} {:.1f} samples/sec \n".format(
                        i_epoch, epochs, i_batch, samples_per_sec
                    )
                )

                i_log += 1

            # Adjust learning rate if needed
            scheduler.step(validation_loss)

            # Display the layer weights
            weights = self.get_layer_weights()
            if weights is not None:
                for i, w in enumerate(weights):
                    self.summary_writer.add_histogram(f"weights_layer_{i}", w, i_log)

        self.log.info("... Done")

    def forward(self, inputs, *kwargs):
        """
        Defines the computation performed at every call.
        Should be overriden by all subclasses.
        """
        raise NotImplementedError