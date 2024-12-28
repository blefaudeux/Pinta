#!/usr/local/bin/python3


import logging
import time
from contextlib import suppress
from itertools import cycle

import torch
import torch.nn as nn
import torch.optim as optim
from pinta.settings import Scheduler, Optimizer, Settings
from pinta.settings import device as _device
from torch.optim.lr_scheduler import CosineAnnealingLR, ReduceLROnPlateau
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
        self.log.setLevel(level=logging.INFO)
        self.output_size = None

        # Set up TensorBoard
        self.summary_writer = SummaryWriter(logdir)

    @property
    def valid(self):
        return self._valid

    def save(self, name):
        with open(name, "wb") as fileio:
            torch.save(self.state_dict(), fileio)

    @staticmethod
    def _split_inputs(inputs: torch.Tensor, settings: Settings):
        """Split the inputs in between the signal and the tuning -slow moving- parts
        Dimensions are [Batch x Channels x TimeSequence]
        """
        return torch.split(
            inputs, [len(settings.inputs), len(settings.tuning_inputs)], dim=1
        )

    def evaluate(self, dataloader: DataLoader):
        #  Re-use PyTorch losses on the fly
        criterion = nn.MSELoss()
        losses = []

        for seq in dataloader:
            seq = seq.to(_device)

            out, _ = self(seq.inputs)
            loss = criterion(out, seq.outputs.view(out.size()[0], -1))
            losses.append(loss.item())

        return losses

    def predict(
        self,
        dataloader: DataLoader,
        mean: torch.Tensor | None = None,
        std: torch.Tensor | None = None,
    ):
        # Move the predictions to cpu() on the fly to save on GPU memory
        predictions = []
        for seq in dataloader:
            predictions.append(self(seq.inputs.to(_device))[0].detach().cpu())
            del seq
        predictions_tensor = torch.cat(predictions).squeeze()

        # De-normalize the output
        if mean is not None and std is not None:
            return torch.add(torch.mul(predictions_tensor, std.cpu()), mean.cpu())

        return predictions_tensor

    def fit(
        self,
        trainer: DataLoader,
        tester: DataLoader,
        settings: Settings,
    ):
        # Setup the training loop
        optimizer = {
            Optimizer.ADAM_W: optim.AdamW(
                self.parameters(),
                lr=settings.training.optim.learning_rate,
                amsgrad=False,
            ),
            Optimizer.SGD: optim.SGD(
                self.parameters(),
                lr=settings.training.optim.learning_rate,
                momentum=settings.training.optim.momentum,
            ),
        }[settings.training.optim.name]

        scheduler = {
            Scheduler.REDUCE_PLATEAU: ReduceLROnPlateau(
                optimizer=optimizer,
                patience=settings.training.optim.scheduler_patience,
                factor=settings.training.optim.scheduler_factor,
            ),
            Scheduler.COSINE: CosineAnnealingLR(
                optimizer=optimizer,
                T_max=settings.training.epoch,
                eta_min=1e-6,
                last_epoch=-1,
            ),
        }[Scheduler(settings.training.optim.scheduler)]

        # FIXME: Handle different losses
        criterion = nn.MSELoss()

        if len(tester) < len(trainer):
            tester = cycle(tester)  # type: ignore

        # If AMP is enabled, create an autocast context. Noop if normal full precision training
        use_bf16 = settings.training.bf16 and _device.type == torch.device("cuda").type
        precision_context = (
            torch.autocast(device_type="cuda", dtype=torch.bfloat16)
            if use_bf16
            else suppress()
        )

        # Now to the actual training
        self.log.info(
            "\nTraining the model... BF16: %s\n" % "enabled" if use_bf16 else "disabled"
        )
        i_log = 0
        for i_epoch in range(settings.training.epoch):
            self.log.info("***** Epoch %d", i_epoch)
            self.log.info(
                " {}/{} LR: {:.4f}".format(
                    i_epoch, settings.training.epoch, optimizer.param_groups[0]["lr"]
                )
            )

            for i_batch, (train_batch, validation_batch) in enumerate(
                zip(trainer, tester)
            ):
                batch_start = time.time()

                # Eval computation on the training data
                optimizer.zero_grad()

                with precision_context:  # type: ignore
                    train_batch = train_batch.to(_device)
                    validation_batch = validation_batch.to(_device)

                    # Split inputs and training inputs, then go through the mode + mixer:
                    if hasattr(self, "tuning_encoder"):
                        inputs, tuning_inputs = self._split_inputs(
                            train_batch.inputs, settings
                        )
                        out = self(inputs, tuning_inputs)

                    else:
                        out, _ = self(train_batch.inputs)
                    loss = criterion(out.squeeze(), train_batch.outputs.squeeze())

                    # Vanilla, backward will populate the gradients and we just step()
                    loss.backward()
                    optimizer.step()

                self.log.info(
                    " {}/{},{} Train loss: {:.4f}".format(
                        i_epoch, settings.training.epoch, i_batch, loss.item()
                    )
                )
                self.summary_writer.add_scalar("train", loss.item(), i_log)

                # Loss on the validation data
                with torch.no_grad(), precision_context:  # type: ignore
                    if hasattr(self, "tuning_encoder"):
                        inputs, tuning_inputs = self._split_inputs(
                            validation_batch.inputs, settings
                        )
                        pred = self(inputs, tuning_inputs)
                    else:
                        pred, _ = self(validation_batch.inputs)
                    validation_loss = criterion(
                        pred.squeeze(), validation_batch.outputs.squeeze()
                    ).detach()

                    self.summary_writer.add_scalar(
                        "validation", validation_loss.item(), i_log
                    )
                    self.log.info(
                        " {}/{},{} Validation loss: {:.4f}".format(
                            i_epoch,
                            settings.training.epoch,
                            i_batch,
                            validation_loss.item(),
                        )
                    )

                # Time some operations
                batch_stop = time.time()
                elapsed = batch_stop - batch_start

                samples_per_sec = (
                    train_batch.inputs.shape[0] + validation_batch.inputs.shape[0]
                ) / elapsed

                self.summary_writer.add_scalar(
                    "Samples_per_sec", samples_per_sec, i_log
                )
                self.summary_writer.add_scalar(
                    "LR", optimizer.param_groups[0]["lr"], i_log
                )

                self.log.info(
                    " {}/{},{} {:.1f}k samples/sec \n".format(
                        i_epoch, settings.training.epoch, i_batch, samples_per_sec / 1e3
                    )
                )

                i_log += 1

                # Adjust learning rate if needed
                if isinstance(scheduler, ReduceLROnPlateau):
                    scheduler.step(metrics=validation_loss)

                # Early stop if no progress
                if optimizer.param_groups[0]["lr"] < 1e-6:
                    self.log.warning("No progress, stopping training")
                    return

            # Report the layer weights / histograms
            for p in self.parameters():
                self.summary_writer.add_histogram(f"weights{p}", p, i_log)

        self.log.info("... Done")

    def forward(self, inputs, tuning_inputs, *kwargs):
        """
        Defines the computation performed at every call.
        Should be overriden by all subclasses.

        Args:
            inputs: fast moving signals, could be about attitude, speed,...
            tuning_inputs: slow moving signals, could be about some boat tuning parameters
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
