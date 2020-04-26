#!/usr/bin/env python3

import argparse
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


def run(args):
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
            len(training_bundle), training_settings["train_batch_size"]
        )
    )

    # Data augmentation / preparation
    transforms: List[Callable] = [
        Normalize(
            mean.to(settings.device, settings.dtype),
            std.to(settings.device, settings.dtype),
        ),
        RandomFlip(
            dimension=training_settings["inputs"].index("wind_angle_y"), odds=0.5
        ),
        RandomFlip(
            dimension=training_settings["inputs"].index("rudder_angle"), odds=0.5
        ),
    ]

    # Train a new model from scratch if need be
    if not dnn.valid:
        log.info("Training a new model, this can take a while")

        trainer, valider = training_bundle.get_dataloaders(
            training_settings["training_ratio"],
            training_settings["seq_length"],
            training_settings["train_batch_size"],
            training_settings["val_batch_size"],
            shuffle=True,
            transforms=transforms,
            dtype=settings.dtype,
            device=settings.device,
        )

        dnn.fit(trainer, valider, settings=training_settings, epochs=EPOCH)
        dnn.save("trained/" + settings.get_name() + ".pt")

    # Check the training
    if args.evaluate is True:
        log.info("Evaluating the model")
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
    if args.plot is True:
        log.info("---\nQuality evaluation:")

        # - de-whiten the data
        def denormalize(data: torch.Tensor):
            return torch.add(
                torch.mul(data, std.to(settings.device, settings.dtype).outputs),
                mean.to(settings.device, settings.dtype).outputs,
            )

        # - prediction: go through the net, split the output sequence to re-align,
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
            denormalize(batch.outputs).detach().cpu().numpy() for batch in tester
        ]

        # - split back to restore the individual datasets
        prediction = np.split(prediction, split_indices)
        reference = np.split(reference, split_indices)

        plt.parallel_plot(
            reference + prediction,
            ["Ground truth" for _ in range(len(reference))]
            + ["Conv" for _ in range(len(prediction))],
            "Network predictions vs ground truth",
        )

    log.info("--Done")


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Generate VPP")
    parser.add_argument(
        "--model_path",
        action="store",
        help="path to a saved model",
        default="trained/" + settings.get_name() + ".pt",
    )

    parser.add_argument(
        "--evaluate",
        action="store_true",
        help="evaluate an existing model, compute error metrics",
    )
    parser.add_argument(
        "--plot",
        action="store_true",
        help="generate a plot to visually compare the ground truth and predictions",
    )

    args = parser.parse_args()
    run(args)
