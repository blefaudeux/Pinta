from __future__ import annotations

# Our lightweight base data structure..
# specialize inputs/outputs, makes it readable down the line
from collections import namedtuple
from typing import Callable, List, Tuple

import numpy as np
import torch
import torchvision
from torch.utils.data import DataLoader, Dataset, random_split

from settings import device, dtype

TrainingSample = namedtuple("TrainingSample", ["inputs", "outputs"])


class TrainingSet(Dataset):
    """
    Holds the training or testing data, with some helper functions.
    This keeps all the time coherent data in one package
    """

    def __init__(self, inputs: torch.Tensor, outputs: torch.Tensor):
        assert inputs.shape[0] == outputs.shape[0], "Dimensions mismatch"

        self.inputs = inputs    # type: torch.Tensor
        self.outputs = outputs  # type: torch.Tensor
        self.transform = lambda x: x

    # Alternative "constructor", straight from Numpy arrays
    @classmethod
    def from_numpy(cls, inputs: np.array, outputs: np.array):
        input_tensor = torch.from_numpy(inputs).to(device=device)
        output_tensor = torch.from_numpy(outputs).to(device=device)

        return cls(input_tensor, output_tensor)

    # Alternative constructor: straight from TrainingSample, repeat
    @classmethod
    def from_training_sample(cls, sample: TrainingSample, seq_len: int):
        inputs = torch.tensor(
            np.repeat(np.array([sample.inputs]), seq_len, axis=0), dtype=dtype
        )
        outputs = torch.tensor(
            np.repeat(np.array([sample.outputs]), seq_len, axis=0), dtype=dtype
        )

        return cls(inputs, outputs)

    def append(self, inputs: torch.Tensor, outputs: torch.Tensor):
        assert inputs.shape[0] == outputs.shape[0], "Dimensions mismatch"

        self.inputs = torch.cat((self.inputs, inputs), 0).to(
            device=device, dtype=dtype)
        self.outputs = torch.cat((self.inputs, inputs), 0).to(
            device=device, dtype=dtype)

    def __getitem__(self, index):
        return self.transform(TrainingSample(inputs=self.inputs[index, :, :],
                                             outputs=self.outputs[index, :]))

    def __len__(self):
        return self.inputs.shape[0]

    def set_transforms(self, transforms: List[Callable]):
        self.transform = torchvision.transforms.Compose(transforms)

    def scale(self, mean: torch.Tensor, std: torch.Tensor):
        self.inputs = torch.mul(
            torch.add(self.inputs, mean[0].reshape(1, -1, 1)),
            std[0].reshape(1, -1, 1))

        self.outputs = torch.mul(
            torch.add(self.outputs, mean[1].reshape(1, -1, 1)),
            std[1].reshape(1, -1, 1))


class TrainingSetBundle:
    """
    Hold a list of training sets, with some helper functions.
    This allows us to maintain a bigger pool of data
    without any time coherency / time continuity constraints.
    All the data can be used for training, but we can enforce
    time-continuous streams where needed.c
    """

    def __init__(self, training_sets: List[TrainingSet]):
        self.sets = training_sets

    def __len__(self):
        return sum(map(lambda x: len(x), self.sets))

    def get_norm(self) -> Tuple[TrainingSample, TrainingSample]:
        """Get Mean and STD over the whole bundle

        Returns:
            [Means],[STD] -- Inputs/Outputs mean and STD
        """

        mean_inputs, mean_outputs, std_inputs, std_outputs = [], [], [], []
        for training_set in self.sets:
            mean_inputs.append(training_set.inputs.mean(dim=0))
            mean_outputs.append(training_set.outputs.mean(dim=0))

            std_inputs.append(training_set.inputs.std(dim=0))
            std_outputs.append(training_set.outputs.std(dim=0))

        # To Torch tensor and mean
        mean = TrainingSample(torch.stack(mean_inputs).mean(
            dim=0), torch.stack(mean_outputs).mean(dim=0))

        std = TrainingSample(torch.stack(std_inputs).mean(
            dim=0), torch.stack(std_outputs).mean(dim=0))

        return mean, std

    def normalize(self):
        """
        Normalize the data, bring it back to zero mean and STD of 1
        """
        mean, std = self.get_norm()

        for training_set in self.sets:
            training_set.inputs = np.subtract(
                training_set.inputs, mean[0]).transpose()
            training_set.inputs = np.divide(
                training_set.inputs, std[0]).transpose()

            training_set.outputs = np.subtract(
                training_set.outputs, mean[1]).transpose()
            training_set.outputs = np.divide(
                training_set.outputs, std[1]).transpose()

    def get_sequences(self, seq_len) -> TrainingSet:
        """
        Prepare sequences of a given length given the input data
        """

        inputs = []
        outputs = []

        for training_set in self.sets:
            a, b = self.generate_temporal_seq(
                training_set.inputs, training_set.outputs, seq_len)
            inputs.append(a)
            outputs.append(b)

        tensor_input = torch.cat(inputs).to(device=device)
        tensor_output = torch.cat(outputs).to(device=device)

        return TrainingSet(tensor_input, tensor_output)

    @staticmethod
    def generate_temporal_seq(tensor_input: torch.Tensor,
                              tensor_output: torch.Tensor,
                              seq_len: int):
        """
        Generate all the subsequences over time,
        Useful for instance for training a temporal conv net
        """

        n_sequences = tensor_input.shape[0] - seq_len + 1

        input_seq = torch.transpose(
            torch.stack([tensor_input[start:start+seq_len, :]
                         for start in range(n_sequences)],
                        dim=0), 1, 2).type(dtype)

        output_seq = tensor_output[:-seq_len+1, :].type(dtype)

        return input_seq, output_seq

    def get_dataloaders(self,
                        ratio: float,
                        seq_len: int,
                        batch_size: int,
                        shuffle: bool) -> Tuple[DataLoader, DataLoader]:
        sequences = self.get_sequences(seq_len)
        train_len = int(ratio * len(sequences))
        test_len = len(sequences) - train_len
        trainer, tester = random_split(sequences, [train_len, test_len])

        return (
            DataLoader(trainer, batch_size=batch_size, shuffle=shuffle),
            DataLoader(tester, batch_size=batch_size, shuffle=shuffle)
        )
