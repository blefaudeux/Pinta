#!/usr/bin/env python3


import logging
from typing import Any, Callable, Dict, List

import torch
from numpy.random import random_sample
from pinta.data_processing.training_set import TrainingSample

LOG = logging.getLogger("Transforms")

EPSILON = 1e-3


def transform_factory(params: Dict[str, Any]) -> List[Callable]:
    "Given the requested serialized transform settings, return the corresponding transform sequence"

    transforms: List[Callable] = []

    for transform_param in params["transforms"]:
        transform_name, transform_args = transform_param[0], transform_param[1]

        def get_normalize():
            return Normalize(*params["data"]["stats"])

        def get_denormalize():
            return Denormalize(*params["data"]["stats"])

        def get_random_flip():
            return RandomFlip(dimensions=[params["inputs"].index(p) for p in transform_args[0]], odds=transform_args[1])

        def get_offset():
            return OffsetInputsOutputs(offset_samples=transform_args[0])

        def get_cut_sequence():
            return CutSequence(inputs_cut=transform_args[0], outputs_cut=transform_args[1])

        transforms.append(
            {
                "denormalize": get_denormalize,
                "normalize": get_normalize,
                "random_flip": get_random_flip,
                "half_precision": HalfPrecision,
                "single_precision": SinglePrecision,
                "offset_inputs_outputs": get_offset,
                "cut_sequence": get_cut_sequence,
            }[transform_name]()
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
                torch.add(sample.inputs, self.mean.inputs),
                self.std.inputs,
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

        LOG.info("Initializing Normalize transform with mean\n{}\nand std\n{}".format(mean, std))

        # Catch noise-free data
        if torch.min(self.std.inputs).item() < EPSILON:
            self.std = TrainingSample(torch.ones_like(self.std.inputs), torch.ones_like(self.std.outputs))
            LOG.warning("Noise-free data detected, skip noise normalization")

    def __call__(self, sample: TrainingSample):
        # Non batched data
        if sample.inputs.shape[0] == 1:
            return TrainingSample(
                inputs=torch.div(
                    torch.add(sample.inputs, -self.mean.inputs),
                    self.std.inputs,
                ),
                outputs=torch.div(torch.add(sample.outputs, -self.mean.outputs), self.std.outputs),
            )

        # Batch data coming in. Could also be handled through broadcasting
        return TrainingSample(
            inputs=torch.div(
                torch.add(sample.inputs, -self.mean.inputs),
                self.std.inputs,
            ),
            outputs=torch.div(
                torch.add(sample.outputs, -self.mean.outputs),
                self.std.outputs,
            ),
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
                inputs[:, d] *= -1

            return TrainingSample(inputs=inputs, outputs=sample.outputs)

        return sample


class SinglePrecision:
    def __init__(self):
        """
        Move the sample to fp32. Noop if it's already of this type
        """
        pass

    def __call__(self, sample: TrainingSample):
        return TrainingSample(inputs=sample.inputs.float(), outputs=sample.outputs.float())


class HalfPrecision:
    def __init__(self):
        """
        Move the sample to fp16. Noop if it's already of this type
        """
        pass

    def __call__(self, sample: TrainingSample):
        return TrainingSample(inputs=sample.inputs.half(), outputs=sample.outputs.half())


class OffsetInputsOutputs:
    def __init__(self, offset_samples: int):
        """Offset the samples from inputs and outputs. Needed for time prediction for instance

        Args:
            offset_samples (int): number of samples by which inputs and outputs should be offset
        """
        self.offset = offset_samples

    def __call__(self, sample: TrainingSample):
        # FIXME not very elegant, there must be a cleaner, branchless way
        if len(sample.inputs.shape) > 1:
            return TrainingSample(inputs=sample.inputs[: -self.offset, :], outputs=sample.outputs[self.offset :, :])

        return TrainingSample(inputs=sample.inputs[: -self.offset], outputs=sample.outputs[self.offset :])


class CutSequence:
    def __init__(self, inputs_cut: int, outputs_cut: int):
        """Given temporal sequences as inputs and outputs, subselect the ones of interest

        Args:
            inputs_cut (int): the number of samples to keep from the input sequence, following python notation.
                (positive from the beginning of the array, negative from the end.
                -3 means keep the 3 last samples
                None -null in json- means keep everything)

            outputs_cut (int): the number of samples to keep from the output sequence, following python notation
                (positive from the beginning of the array, negative from the end)

        """
        self.inputs_cut = inputs_cut
        self.outputs_cut = outputs_cut

    @staticmethod
    def __cut(seq: torch.Tensor, cut: int):
        if cut is None or cut > 0:
            return seq[:cut, :]

        return seq[cut:, :]

    def __call__(self, sample: TrainingSample):
        return TrainingSample(
            inputs=self.__cut(sample.inputs, self.inputs_cut), outputs=self.__cut(sample.outputs, self.outputs_cut)
        )
