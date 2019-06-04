import torch
import numpy as np


"""
Holds the training or testing data, with some helper functions
"""


class TrainingSet:
    def __init__(self, inputs, outputs):
        if isinstance(inputs, np.array):
            assert inputs.shape[0] == outputs.shape[0], "Dimensions mismatch"
        self.inputs = inputs
        self.outputs = outputs
        self.is_torch = isinstance(inputs, torch.Tensor)

    def to_torch(self):
        if not self.is_torch:
            self.inputs = torch.from_numpy(self.inputs)
            self.outputs = torch.from_numpy(self.inputs)

        self.is_torch = True

    def randomize(self):
        if not self.is_torch:
            self.to_torch()

        shuffle = torch.randperm(self.inputs.shape[0])
        self.inputs = self.input[shuffle]
        self.outputs = self.outputs[shuffle]

    def scale(self, mean, std):
        self.input = torch.mul(
            torch.add(self.input, mean[0].reshape(1, -1, 1)),
            std[0].reshape(1, -1, 1))

        self.output = torch.mul(
            torch.add(self.output, mean[1].reshape(1, -1, 1)),
            std[1].reshape(1, -1, 1))
