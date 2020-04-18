#!/usr/bin/env python3

import logging
from datetime import datetime
from pathlib import Path
from typing import Callable, List

import numpy as np
import torch

import settings
from data_processing import plot as plt
from data_processing.load import load_folder, load_sets
from data_processing.training_set import TrainingSetBundle
from data_processing.transforms import Normalize, RandomFlip
from train.engine_cnn import Conv

# Basic setup: get config and logger
training_settings = settings.get_defaults()
logging.basicConfig(level=logging.INFO)
log = logging.getLogger(training_settings["log"])

#  ConvNN
EPOCH = training_settings["epoch"]
INPUT_SIZE = [len(training_settings["inputs"]), training_settings["seq_length"]]

dnn = Conv(
    logdir="logs/" + settings.get_name() + str(datetime.now()),
    input_size=INPUT_SIZE,
    hidden_size=training_settings["hidden_size"],
    filename="trained/" + settings.get_name() + ".pt",
    log_channel="DNN  ",
)
dnn.to(settings.device)


# Load the dataset
data_list = load_sets(load_folder(Path("data")), training_settings)
training_bundle = TrainingSetBundle(data_list)
mean, std = training_bundle.get_norm()

log.info(
    "Loaded {} samples. Batch is {}".format(
        len(training_bundle), training_settings["batch_size"]
    )
)

# Data augmentation / preparation
transforms: List[Callable] = [
    Normalize(
        mean.to(settings.device, settings.dtype),
        std.to(settings.device, settings.dtype),
    ),
    RandomFlip(dimension=training_settings["inputs"].index("wind_angle_x"), odds=0.5),
    RandomFlip(dimension=training_settings["inputs"].index("rudder_angle"), odds=0.5),
]

# Train a new model from scratch if need be
if not dnn.valid:
    trainer, valider = training_bundle.get_dataloaders(
        training_settings["training_ratio"],
        training_settings["seq_length"],
        training_settings["batch_size"],
        shuffle=True,
        transforms=transforms,
        dtype=settings.dtype,
        device=settings.device,
    )

    dnn.fit(trainer, valider, settings=training_settings, epochs=EPOCH)
    dnn.save("trained/" + settings.get_name() + ".pt")

# Check the training
tester, split_indices = training_bundle.get_sequential_dataloader(
    training_settings["seq_length"],
    transforms=[
        Normalize(
            mean.to(settings.device, settings.dtype),
            std.to(settings.device, settings.dtype),
        )
    ],
    dtype=settings.dtype,
    device=settings.device,
)
losses = dnn.evaluate(tester, training_settings)
log.info("Final test Score: %.2f RMSE" % np.sqrt(sum(losses) / len(losses)))


# Compare visually the outputs
# - de-whiten the data
def denormalize(data: torch.Tensor):
    return torch.add(
        torch.mul(data, std.to(settings.device, settings.dtype).outputs),
        mean.to(settings.device, settings.dtype).outputs,
    )


# - prediction: go through the net, split the output sequence to re-align,
log.info("---\nQuality evaluation:")

prediction = (
    dnn.predict(
        tester,
        mean=mean.to(settings.device, settings.dtype).outputs,
        std=std.to(settings.device, settings.dtype).outputs,
    )
    .detach()
    .cpu()
    .numpy()
)

reference = [
    denormalize(batch.outputs[: -training_settings["seq_length"] + 1])
    .detach()
    .cpu()
    .numpy()
    for batch in tester
]

# - split back to restore the individual datasets
prediction = np.split(prediction, split_indices)
reference = np.split(reference, split_indices)

plt.parallel_plot(
    reference + prediction,
    ["Ground truth" for _ in range(len(tester))]
    + ["Conv" for _ in range(len(prediction))],
    "Network predictions vs ground truth",
)

log.info("--Done")
