#!/usr/local/bin/python3
"""
Implement different NNs which best describe the behaviour of the system
"""

import logging
import time
from contextlib import suppress
from itertools import cycle
from typing import Any, Dict, Union

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR, ReduceLROnPlateau
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from settings import Scheduler
from settings import device as _device
from settings import dtype as _dtype

try:
    from torch.cuda.amp import autocast, GradScaler

    _amp_available = True
except ImportError:
    _amp_available = False


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
            seq = seq.to(_device, _dtype)

            out, _ = self(seq.inputs)
            loss = criterion(out, seq.outputs.view(out.size()[0], -1))
            losses.append(loss.item())

        return losses

    def predict(
        self, dataloader: DataLoader, mean: torch.Tensor = None, std: torch.Tensor = None,
    ):
        # Move the predictions to cpu() on the fly to save on GPU memory
        predictions = [self(seq.inputs.to(_device, _dtype))[0].detach().cpu() for seq in dataloader]
        predictions_tensor = torch.cat(predictions).squeeze()

        # De-normalize the output
        if mean and std:
            return torch.add(torch.mul(predictions_tensor, std.cpu()), mean.cpu())

        return predictions_tensor

    def fit(
        self, trainer: DataLoader, tester: DataLoader, settings: Dict[str, Any], epochs: int = 50,
    ):
        # Setup the training loop
        optimizer = optim.Adam(self.parameters(), lr=settings["learning_rate"], amsgrad=True)

        scheduler: Union[ReduceLROnPlateau, CosineAnnealingLR] = ReduceLROnPlateau(
            optimizer=optimizer, patience=2
        ) if settings["scheduler"] == Scheduler.REDUCE_PLATEAU else CosineAnnealingLR(
            optimizer=optimizer, T_max=epochs, eta_min=1e-6, last_epoch=-1
        )

        criterion = nn.MSELoss()

        if len(tester) < len(trainer):
            tester = cycle(tester)  # type: ignore

        # If AMP is enabled, create an autocast context. Noop if normal full precision training
        use_amp = settings["amp"] and _device.type == torch.device("cuda").type and _amp_available
        context = autocast() if use_amp else suppress()
        scaler = GradScaler() if use_amp else None

        # Now to the actual training
        self.log.info("Training the network...\n")
        i_log = 0
        for i_epoch in range(epochs):

            self.log.info("***** Epoch %d", i_epoch)
            self.log.info(" {}/{} LR: {:.4f}".format(i_epoch, epochs, optimizer.param_groups[0]["lr"]))

            validation_loss = torch.zeros(1)

            for i_batch, (train_batch, validation_batch) in enumerate(zip(trainer, tester)):
                batch_start = time.time()

                # Eval computation on the training data
                optimizer.zero_grad()
                train_batch = train_batch.to(_device, _dtype)

                # FW pass, optionally with mixed precision
                with context:
                    out, _ = self(train_batch.inputs)
                    loss = criterion(out.squeeze(), train_batch.outputs.squeeze())

                if scaler is not None:
                    # AMP training path
                    # The scaler will make sure that the fp16 grads don't underflow
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    # Vanilla, backward will populate the gradients and we just step()
                    loss.backward()
                    optimizer.step()

                self.log.info(" {}/{},{} Train loss: {:.4f}".format(i_epoch, epochs, i_batch, loss.item()))
                self.summary_writer.add_scalar("train", loss.item(), i_log)

                # Loss on the validation data
                def closure_validation(data=validation_batch):
                    data = data.to(_device, _dtype)
                    pred, _ = self(data.inputs)
                    loss = criterion(pred.squeeze(), data.outputs.squeeze()).detach()

                    self.summary_writer.add_scalar("validation", loss.item(), i_log)
                    self.log.info(" {}/{},{} Validation loss: {:.4f}".format(i_epoch, epochs, i_batch, loss.item()))
                    return loss

                validation_loss += closure_validation()

                # Time some operations
                batch_stop = time.time()
                elapsed = batch_stop - batch_start

                samples_per_sec = (train_batch.inputs.shape[0] + validation_batch.inputs.shape[0]) / elapsed

                self.summary_writer.add_scalar("Samples_per_sec", samples_per_sec, i_log)
                self.summary_writer.add_scalar("LR", optimizer.param_groups[0]["lr"], i_log)

                self.log.info(" {}/{},{} {:.1f} samples/sec \n".format(i_epoch, epochs, i_batch, samples_per_sec))

                i_log += 1

            # Adjust learning rate if needed
            if isinstance(scheduler, ReduceLROnPlateau):
                scheduler.step(metrics=validation_loss, epoch=i_epoch)

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

    def load(self, filename):
        try:
            with open(filename, "rb") as f:
                self.load_state_dict(torch.load(f, map_location=_device))
                self.log.info("---\nNetwork {} loaded".format(filename))
                self.log.info(self)
                return True

        except (ValueError, OSError, IOError, TypeError) as exception:
            self.log.warning(exception)
            self.log.warning("Could not find or load existing NN")
            return False
