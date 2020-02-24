#!/usr/bin/env python3


import torch

from .training_set import TrainingSample


class Denormalize:
    def __init__(self, mean: TrainingSample, std: TrainingSample):
        self.mean = mean
        self.std = std

    def __call__(self, sample: TrainingSample):

        return TrainingSample(
            inputs=torch.mul(
                torch.add(sample.inputs, self.mean.inputs.reshape(1, -1, 1)),
                self.std.inputs.reshape(1, -1, 1)),
            outputs=torch.mul(
                torch.add(sample.outputs, self.mean.outputs), self.std.outputs))


class Normalize:
    def __init__(self, mean: TrainingSample, std: TrainingSample):
        """
        Given mean and std of the appropriate dimensions,
        bring the training set to a normalized state

        Arguments:
            mean {TrainingSample holding torch tensors} -- Expected distribution first moment
            std {TrainingSample holding torch tensors} -- Expected distribution second moment
        """
        self.mean = mean
        self.std = std

    def __call__(self, sample: TrainingSample):
        return TrainingSample(
            inputs=torch.div(
                torch.add(sample.inputs, - self.mean.inputs.reshape(1, -1, 1)),
                self.std.inputs.reshape(1, -1, 1)),
            outputs=torch.div(
                torch.add(sample.outputs, - self.mean.outputs), self.std.outputs))
