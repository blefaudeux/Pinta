#!/usr/bin/env python3

import logging
from pathlib import Path

import numpy as np
import torch

from pinta.data_processing import plot as plt
from pinta.data_processing.load import load_folder, load_sets
from pinta.data_processing.training_set import TrainingSetBundle
from pinta.data_processing.transforms import (
    Normalize,
    OffsetInputsOutputs,
    transform_factory,
)
from pinta.model.model_factory import model_factory
import hydra
from omegaconf import DictConfig
import os


@hydra.main(config_path="configs", config_name="imoca_polar")
def run(cfg: DictConfig):
    logging.basicConfig(level=logging.INFO)
    log = logging.getLogger(cfg.log)
    log.setLevel(level=logging.INFO)

    if not cfg.model.path:
        # Generate a default name which describes the settings
        cfg.model.path = (
            "./"
            + cfg.model.model_type
            + f"_seq_{cfg.model.seq_length}"
            + f"_hidden_{cfg.model.hidden_size}"
            + f"_batch_{cfg.data.train_batch_size}"
            + f"_lr_{cfg.training.optim.learning_rate}"
            + f"_ep_{cfg.training.epoch}"
            + ".pt"
        )

    if not cfg.data.path:
        cfg.data.path = os.environ.get("DATA_PATH")

    log.info("Settings:\n{}".format(cfg))
    model = model_factory(cfg, model_path=cfg.model.path)

    # Load the dataset
    dataframes = load_folder(
        Path(cfg.data.path),
        zero_mean_helm=False,
        parallel_load=cfg.data.get("parallel_load", True),
        max_number_sequences=cfg.data.get("max_number_sequences", -1),
    )
    data_list = load_sets(dataframes, cfg)
    training_bundle = TrainingSetBundle(data_list)

    if cfg.data.get("statistics", None) is None:
        log.info("Updating the normalization statistics with the current data pool")
        statistics = training_bundle.get_norm()
    else:
        statistics = cfg.data.statistics

    log.info(
        "Loaded {} samples. Batch is {}".format(
            len(training_bundle), cfg.data.train_batch_size
        )
    )
    log.info("Available fields: {}".format(dataframes[0].columns.values))

    # Data augmentation / preparation.
    transforms = transform_factory(cfg, statistics)

    # Adjust for a possible time offset requirement
    offset_transform = list(
        filter(lambda t: isinstance(t, OffsetInputsOutputs), transforms)
    )

    if len(offset_transform) > 0:
        offset = offset_transform.pop().offset
        log.info(
            "Offset transform requested, adjusting the raw sequence length to {}".format(
                cfg.model.seq_length + offset
            )
        )
        cfg.model.seq_length += offset

    # Train a new model from scratch if need be
    if not model.valid or cfg.force_new:
        log.info("Training a new model, this can take a while")
        training_set, validation_set = training_bundle.get_dataloaders(
            cfg,
            transforms=transforms,
        )

        model.fit(training_set, validation_set, settings=cfg)
        model.save(cfg.model.path)

    if cfg.get("evaluate", False) or cfg.get("plot", False):
        # Generate linear test data
        tester, split_indices = training_bundle.get_sequential_dataloader(
            cfg.model.seq_length,
            transforms=[Normalize(*statistics)],
            batch_size=cfg.data.train_batch_size,
        )

        # # Check the training
        # if cfg.evaluate:
        #     log.info("Evaluating the model")

        #     losses = model.evaluate(tester)
        #     log.info("Final test Score: %.2f RMSE" % np.sqrt(sum(losses) / len(losses)))

        # Compare visually the outputs
        if cfg.plot:
            # FIXME: @lefaudeux Timings are misaligned
            log.info("---\nQuality evaluation:")
            mean, std = statistics

            # - prediction: go through the net, split the output sequence to re-align,
            prediction = (
                model.predict(
                    tester,
                    mean=mean.outputs,
                    std=std.outputs,
                )
                .detach()
                .cpu()
                .numpy()
            )

            # - de-whiten the data
            std, mean = std.to(cfg.device), mean.to(cfg.device)

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
    run()
