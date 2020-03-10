#!/usr/bin/env python3


import logging

import torch
from numpy.random import random_sample

from .training_set import TrainingSample

LOG = logging.getLogger("Transforms")


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

        LOG.info(
            "Initializing De-Normalize transform with mean {} and std {}".format(
                mean, std
            )
        )

    def __call__(self, sample: TrainingSample):

        return TrainingSample(
            inputs=torch.mul(
                torch.add(sample.inputs, self.mean.inputs.reshape(1, -1, 1)),
                self.std.inputs.reshape(1, -1, 1),
            ),
            outputs=torch.mul(
                torch.add(sample.outputs, self.mean.outputs), self.std.outputs
            ),
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

        LOG.info(
            "Initializing Normalize transform with mean {} and std {}".format(mean, std)
        )

    def __call__(self, sample: TrainingSample):
        return TrainingSample(
            inputs=torch.div(
                torch.add(sample.inputs, -self.mean.inputs.reshape(1, -1, 1)),
                self.std.inputs.reshape(1, -1, 1),
            ),
            outputs=torch.div(
                torch.add(sample.outputs, -self.mean.outputs), self.std.outputs
            ),
        )


class RandomFlip:
    def __init__(
        self, dimension: int, odds: float,
    ):
        """
        Randomly flip the given dimension.

        Arguments:
            odds {float}
                -- odds [0,1] of the flip happening
            dimension {int}
                -- which dimension should be flipped
        """
        self.odds = odds
        self.dim = dimension
        LOG.info(
            "Initializing Random flip transform on dimension {} with odds {}".format(
                dimension, odds
            )
        )

    def __call__(self, sample: TrainingSample):
        if random_sample() < self.odds:
            inputs = sample.inputs
            inputs[0, self.dim, :] *= -1
            return TrainingSample(inputs=inputs, outputs=sample.outputs)

        return sample
