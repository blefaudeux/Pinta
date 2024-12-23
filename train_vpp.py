#!/usr/bin/env python3

import argparse
import logging
from pathlib import Path

import numpy as np
import torch

import pinta.settings as settings
from pinta.data_processing import plot as plt
from pinta.data_processing.load import load_folder, load_sets
from pinta.data_processing.training_set import TrainingSetBundle
from pinta.data_processing.transforms import (
    Normalize,
    OffsetInputsOutputs,
    SinglePrecision,
    transform_factory,
)
from pinta.model.model_factory import model_factory


def run(args):
    # Basic setup: get config and logger
    params = settings.load(args.settings_path)

    logging.basicConfig(level=logging.INFO)
    log = logging.getLogger(params.log)
    log.setLevel(level=logging.INFO)

    log.info("Settings:\n{}".format(params))

    if not args.model_path:
        # Generate a default name which describes the settings
        args.model_path = "trained/" + str(params) + ".pt"
    dnn = model_factory(params, model_path=args.model_path)

    # Load the dataset
    dataframes = load_folder(
        Path(args.data_path),
        zero_mean_helm=False,
        parallel_load=args.parallel,
        max_number_sequences=args.max_number_sequences,
    )
    data_list = load_sets(dataframes, params)
    training_bundle = TrainingSetBundle(data_list)

    if params.data.statistics is None:
        log.info("Updating the normalization statistics with the current data pool")
        params.data.statistics = training_bundle.get_norm()

    log.info(
        "Loaded {} samples. Batch is {}".format(
            len(training_bundle), params.data.train_batch_size
        )
    )
    log.info("Available fields: {}".format(dataframes[0].columns.values))

    # Data augmentation / preparation.
    transforms = transform_factory(params)

    # Adjust for a possible time offset requirement
    offset_transform = list(
        filter(lambda t: isinstance(t, OffsetInputsOutputs), transforms)
    )

    if len(offset_transform) > 0:
        offset = offset_transform.pop().offset
        log.info(
            "Offset transform requested, adjusting the raw sequence length to {}".format(
                params.trunk.seq_length + offset
            )
        )
        params.trunk.seq_length += offset

    # Train a new model from scratch if need be
    if not dnn.valid:
        log.info("Training a new model, this can take a while")
        training_set, validation_set = training_bundle.get_dataloaders(
            params,
            transforms=transforms,
        )

        dnn.fit(training_set, validation_set, settings=params)
        dnn.save(args.model_path)

    if args.evaluate or args.plot:
        # Generate linear test data
        tester, split_indices = training_bundle.get_sequential_dataloader(
            params["seq_length"],
            transforms=[Normalize(*params.data.stats), SinglePrecision()],
            batch_size=params.data.train_batch_size,
        )

        # Check the training
        if args.evaluate:
            log.info("Evaluating the model")

            losses = dnn.evaluate(tester, params)
            log.info("Final test Score: %.2f RMSE" % np.sqrt(sum(losses) / len(losses)))

        # Compare visually the outputs
        if args.plot:
            # FIXME: @lefaudeux Timings are misaligned
            log.info("---\nQuality evaluation:")
            mean, std = params.data.statistics

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

            # - de-whiten the data
            std, mean = std.to(settings.device), mean.to(settings.device)

            def denormalize(data: torch.Tensor):
                # Broadcast the normalization factor to the hidden dimensions
                return torch.add(
                    torch.mul(data, std.outputs),
                    mean.outputs,
                )

            reference = (
                torch.cat([denormalize(batch.outputs[:, :, -1]) for batch in tester])
                .detach()
                .cpu()
                .numpy()
            )

            # - limit the display to fewer samples
            SAMPLES = 500
            reference = reference[:SAMPLES]
            prediction = prediction[:SAMPLES]

            # - split back to restore the individual datasets
            if len(split_indices) > 1:
                prediction = np.split(prediction, split_indices)
                reference = np.split(reference, split_indices)
            else:
                prediction = [prediction]
                reference = [reference]

            plt.parallel_plot(
                reference + prediction,
                [f"Ground truth {i}" for i in range(len(reference))]
                + [f"Prediction {i}" for i in range(len(prediction))],
                title="Network predictions vs ground truth",
                auto_open=True,
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
        "--settings_path",
        action="store",
        help="path to the json settings for the run",
        default=None,
    )

    parser.add_argument(
        "--plot",
        action="store_true",
        help="generate a plot to visually compare the ground truth and predictions",
    )

    parser.add_argument(
        "--parallel",
        action="store_true",
        help="Load muliple files in parallel. Errors may not be properly visible",
        default=True,
    )

    parser.add_argument(
        "--evaluate",
        action="store_true",
        help="Test a model against a new dataset",
        default=False,
    )

    parser.add_argument(
        "--max_number_sequences",
        action="store",
        help="Optionally limit the number of sequences to load from a data pool",
        default=-1,
        type=int,
    )

    args = parser.parse_args()
    run(args)
