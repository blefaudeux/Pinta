from __future__ import annotations

import torch
import numpy as np
from typing import List, Tuple

"""
Holds the training or testing data, with some helper functions
"""


class TrainingSet:
    def __init__(self, inputs: torch.Tensor, outputs: torch.Tensor):
        assert inputs.shape[0] == outputs.shape[0], "Dimensions mismatch"

        self.inputs = inputs    # type: torch.Tensor
        self.outputs = outputs  # type: torch.Tensor

    # Alternative "constructor", straight from Numpy arrays
    @classmethod
    def from_numpy(cls, inputs: np.array, outputs: np.array):
        return cls(torch.from_numpy(inputs), torch.from_numpy(outputs))

    def append(self, inputs: torch.Tensor, outputs: torch.Tensor):
        assert inputs.shape[0] == outputs.shape[0], "Dimensions mismatch"

        self.inputs = torch.cat((self.inputs, inputs), 0)
        self.outputs = torch.cat((self.inputs, inputs), 0)

    def randomize(self):
        shuffle = torch.randperm(self.inputs.shape[0])
        self.inputs = self.input[shuffle]
        self.outputs = self.outputs[shuffle]

    def get_train_test(self, ratio: float, randomize: bool) -> Tuple[TrainingSet, TrainingSet]:
        """
        Return two training sets, either randomly selected or sequential

        Arguments:
            ratio {float} -- train/test ratio
        """

        len_training_set = round(self.inputs.shape[0] * ratio)

        if randomize:
            index = torch.randperm(self.inputs.shape[0])
        else:
            index = np.arange(self.inputs.shape[0])

        return (TrainingSet(self.inputs[index[:len_training_set]],
                            self.outputs[index[:len_training_set]]),
                TrainingSet(self.inputs[index[len_training_set:]],
                            self.outputs[index[len_training_set:]]))

    def scale(self, mean: torch.Tensor, std: torch.Tensor):
        self.input = torch.mul(
            torch.add(self.inputs, mean[0].reshape(1, -1, 1)),
            std[0].reshape(1, -1, 1))

        self.output = torch.mul(
            torch.add(self.outputs, mean[1].reshape(1, -1, 1)),
            std[1].reshape(1, -1, 1))
