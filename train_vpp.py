#!/usr/bin/env python3

import argparse
import logging
from pathlib import Path
from typing import Callable, List

import numpy as np
import torch

import settings
from data_processing import plot as plt
from data_processing.load import load_folder, load_sets
from data_processing.training_set import TrainingSetBundle
from data_processing.transforms import Normalize
from model.model_factory import model_factory


def run(args):
    # Basic setup: get config and logger
    params = settings.get_default_params()
    params["amp"] = args.amp
    logging.basicConfig(level=logging.INFO)
    log = logging.getLogger(params["log"])

    EPOCH = params["epoch"]
    if not args.model_path:
        # Generate a default name which describes the settings
        args.model_path = "trained/" + settings.get_name(params) + ".pt"
    dnn = model_factory(params, model_path=args.model_path)

    # Load the dataset
    dataframes = load_folder(Path(args.data_path), zero_mean_helm=False, parallel_load=args.parallel)
    data_list = load_sets(dataframes, params)
    training_bundle = TrainingSetBundle(data_list)
    mean, std = training_bundle.get_norm()

    log.info("Loaded {} samples. Batch is {}".format(len(training_bundle), params["train_batch_size"]))

    # Data augmentation / preparation
    transforms: List[Callable] = [
        Normalize(
            mean,
            std,
        ),
        # RandomFlip(dimension=[params["inputs"].index("helm"), params["inputs"].index("twa_y")], odds=0.5),
    ]

    # Train a new model from scratch if need be
    if not dnn.valid:
        log.info("Training a new model, this can take a while")

        trainer, valider = training_bundle.get_dataloaders(
            params["training_ratio"],
            params["seq_length"],
            params["train_batch_size"],
            params["val_batch_size"],
            shuffle=True,
            transforms=transforms,
        )

        dnn.fit(trainer, valider, settings=params, epochs=EPOCH)
        dnn.save(args.model_path)

    if args.evaluate or args.plot:
        tester, split_indices = training_bundle.get_sequential_dataloader(
            params["seq_length"],
            transforms=[
                Normalize(
                    mean,
                    std,
                )
            ],
            batch_size=params["train_batch_size"],
        )

    # Check the training
    if args.evaluate:
        log.info("Evaluating the model")

        losses = dnn.evaluate(tester, params)
        log.info("Final test Score: %.2f RMSE" % np.sqrt(sum(losses) / len(losses)))

    # Compare visually the outputs
    if args.plot:
        log.info("---\nQuality evaluation:")

        # - de-whiten the data
        def denormalize(data: torch.Tensor):
            return torch.add(
                torch.mul(data, std.outputs),
                mean.outputs,
            )

        # - prediction: go through the net, split the output sequence to re-align,
        prediction = (
            dnn.predict(
                tester,
                mean=mean.outputs,
                std=std.outputs,
            )
            .detach()
            .cpu()
            .numpy()
        )

        reference = torch.cat([denormalize(batch.outputs) for batch in tester]).detach().cpu().numpy()

        # - split back to restore the individual datasets
        if len(split_indices) > 1:
            prediction = np.split(prediction, split_indices)
            reference = np.split(reference, split_indices)
        else:
            prediction = [prediction]
            reference = [reference]

        plt.parallel_plot(
            reference + prediction,
            [f"Ground truth {i}" for i in range(len(reference))] + [f"Conv {i}" for i in range(len(prediction))],
            "Network predictions vs ground truth",
        )

    log.info("--Done")


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Generate VPP")

    parser.add_argument(
        "--data_path",
        action="store",
        help="path to the training data",
        default="data",
    )

    parser.add_argument(
        "--model_path",
        action="store",
        help="path to a saved model",
        default=None,
    )

    parser.add_argument(
        "--plot",
        action="store_true",
        help="generate a plot to visually compare the ground truth and predictions",
    )

    parser.add_argument(
        "--amp",
        action="store_true",
        help="enable Pytorch Automatic Mixed Precision",
    )

    parser.add_argument(
        "--parallel",
        action="store_true",
        help="Load muliple files in parallel. Errors may not be properly visible",
        default=False,
    )

    parser.add_argument(
        "--evaluate",
        action="store_true",
        help="Test a model against a new dataset",
        default=False,
    )

    args = parser.parse_args()
    run(args)
