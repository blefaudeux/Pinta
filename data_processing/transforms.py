#!/usr/bin/env python3


import logging
from typing import Any, Callable, Dict, List

import torch
from numpy.random import random_sample

from .training_set import TrainingSample

LOG = logging.getLogger("Transforms")

EPSILON = 1e-3


def transform_factory(params: Dict[str, Any]) -> List[Callable]:
    "Given the requested serialized transform settings, return the corresponding transform sequence"

    transforms: List[Callable] = []

    for transform_param in params["transforms"]:
        transform_name = transform_param[0]
        transform_args = transform_param[1]

        def get_normalize():
            return Normalize(*params["data_stats"])

        def get_denormalize():
            return Denormalize(*params["data_stats"])

        def get_random_flip():
            return RandomFlip(dimensions=[params["inputs"].index(p) for p in transform_args[0]], odds=transform_args[1])

        transforms.append(
            {"denormalize": get_denormalize, "normalize": get_normalize, "random_flip": get_random_flip}[
                transform_name
            ]()
        )

    return transforms


"""
.. warning:
    if you add a transform down below (aka data augmentation),
    make sure that it's properly handled by the transform factory just above
"""


class Denormalize:
    def __init__(self, mean: TrainingSample, std: TrainingSample):
        """
        Given mean and std of the appropriate dimensions,
        restore the training set to the original, de-normalized state

        Arguments:
            mean {TrainingSample holding torch tensors}
                -- Expected distribution first moment
            std {TrainingSample holding torch tensors}
                -- Expected distribution second moment
        """
        self.mean = mean
        self.std = std

        LOG.info("Initializing De-Normalize transform with mean:\n{}\nand std\n{}".format(mean, std))

        # Catch noise-free data
        if torch.min(self.std.inputs).item() < EPSILON:
            self.std = TrainingSample(torch.ones_like(self.std.inputs), torch.ones_like(self.std.outputs))
            LOG.warning("Noise-free data detected, skip noise normalization")

    def __call__(self, sample: TrainingSample):
        return TrainingSample(
            inputs=torch.mul(
                torch.add(sample.inputs, self.mean.inputs.reshape(1, -1, 1)),
                self.std.inputs.reshape(1, -1, 1),
            ),
            outputs=torch.mul(torch.add(sample.outputs, self.mean.outputs), self.std.outputs),
        )


class Normalize:
    def __init__(self, mean: TrainingSample, std: TrainingSample):
        """
        Given mean and std of the appropriate dimensions,
        bring the training set to a normalized state

        Arguments:
            mean {TrainingSample holding torch tensors}
                -- Expected distribution first moment
            std {TrainingSample holding torch tensors}
                -- Expected distribution second moment
        """
        self.mean = mean
        self.std = std

        LOG.info("Initializing Normalize transform with mean\n{} and std\n{}".format(mean, std))

        # Catch noise-free data
        if torch.min(self.std.inputs).item() < EPSILON:
            self.std = TrainingSample(torch.ones_like(self.std.inputs), torch.ones_like(self.std.outputs))
            LOG.warning("Noise-free data detected, skip noise normalization")

    def __call__(self, sample: TrainingSample):
        # Non batched data
        if sample.inputs.shape[0] == 1:
            return TrainingSample(
                inputs=torch.div(
                    torch.add(sample.inputs, -self.mean.inputs.reshape(1, -1)),
                    self.std.inputs.reshape(1, -1),
                ),
                outputs=torch.div(torch.add(sample.outputs, -self.mean.outputs), self.std.outputs),
            )

        # Batch data coming in. Could also be handled through broadcasting
        return TrainingSample(
            inputs=torch.div(
                torch.add(sample.inputs, -self.mean.inputs.reshape(1, -1, 1)),
                self.std.inputs.reshape(1, -1, 1),
            ),
            outputs=torch.div(torch.add(sample.outputs, -self.mean.outputs), self.std.outputs),
        )


class RandomFlip:
    def __init__(
        self,
        dimensions: List[int],
        odds: float,
    ):
        """
        Randomly flip the given dimensions.

        Arguments:
            odds {float}
                -- odds [0,1] of the flip happening
            dimension {List[int]}
                -- which dimensions should be flipped
        """
        self.odds = odds
        self.dims = dimensions
        LOG.info("Initializing Random flip transform on dimensions {} with odds {}".format(dimensions, odds))

    def __call__(self, sample: TrainingSample):
        if random_sample() < self.odds:
            # Flip all the dimensions at the same time
            inputs = sample.inputs
            for d in self.dims:
                inputs[0, d, :] *= -1

            return TrainingSample(inputs=inputs, outputs=sample.outputs)

        return sample
