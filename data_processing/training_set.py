from __future__ import annotations

import torch
import numpy as np
from typing import List, Tuple

dtype = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor


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


class TrainingSetBundle:
    def __init__(self, training_sets: List[TrainingSet]):
        self.sets = training_sets

    def get_norm(self):
        """Get Mean and STD over the whole bundle

        Returns:
            [Means],[STD] -- Inputs/Outputs mean and STD
        """

        mean_inputs, mean_outputs, std_inputs, std_outputs = [], [], [], []
        for t in self.sets:
            mean_inputs.append(t.inputs.mean(dim=0))
            mean_outputs.append(t.outputs.mean(dim=0))

            std_inputs.append(t.inputs.std(dim=0))
            std_outputs.append(t.outputs.std(dim=0))

        # To Torch tensor and mean
        mean = [torch.stack(mean_inputs).mean(dim=0), torch.stack(mean_outputs).mean(dim=0)]
        std = [torch.stack(std_inputs).mean(dim=0), torch.cat(std_outputs).mean(dim=0)]

        return mean, std

    def normalize(self):
        """
        Normalize the data, bring it back to zero mean and STD of 1
        """
        mean, std = self.get_norm()

        for training_set in self.sets:
            training_set.inputs = np.subtract(training_set.inputs, mean[0]).transpose()
            training_set.inputs = np.divide(training_set.inputs, std[0]).transpose()

            training_set.outputs = np.subtract(training_set.outputs, mean[1]).transpose()
            training_set.outputs = np.divide(training_set.outputs, std[1]).transpose()

    @staticmethod
    def generate_temporal_seq(input, output, seq_len):
        """
        Generate all the subsequences over time for the conv training
        """

        # TODO: move to pure torch
        n_sequences = input.shape[1] - seq_len + 1

        input_seq = np.array([input[:, start:start+seq_len]
                              for start in range(n_sequences)])

        output_seq = np.array(output[:-seq_len+1, :])

        return torch.from_numpy(input_seq).type(dtype), torch.from_numpy(output_seq).type(dtype)

    def get_sequences(self, seq_len) -> TrainingSet:
        """
        Prepare sequences of a given length given the input data
        """

        inputs = []
        outputs = []

        for trainingSet in self.sets:
            a, b = self.generate_temporal_seq(trainingSet.inputs, trainingSet.outputs, seq_len)
            inputs.append(a)
            outputs.append(b)

        return TrainingSet(torch.cat(inputs), torch.cat(outputs))
