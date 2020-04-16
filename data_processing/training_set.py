from __future__ import annotations

# Our lightweight base data structure..
# specialize inputs/outputs, makes it readable down the line
from collections import namedtuple
from typing import Callable, List, Tuple

import numpy as np
import torch
import torchvision
from torch.utils.data import DataLoader, Dataset, SequentialSampler, random_split

import settings

TrainingSample_base = namedtuple("TrainingSample", ["inputs", "outputs"])


class TrainingSample(TrainingSample_base):
    """
    Holds one or many training samples. Make sure that the dimensions
    and matching in between inputs and outputs are always respected
    """

    def to(
        self, device: torch.device = None, dtype: torch.dtype = None
    ) -> TrainingSample:
        return TrainingSample(
            inputs=self.inputs.to(device, dtype), outputs=self.outputs.to(device, dtype)
        )


class TrainingSet(Dataset):
    """
    Holds the training or testing data, with some helper functions.
    This keeps all the time coherent data in one package
    """

    def __init__(self, inputs: torch.Tensor, outputs: torch.Tensor):
        assert inputs.shape[0] == outputs.shape[0], "Dimensions mismatch"

        self.inputs = inputs  # type: torch.Tensor
        self.outputs = outputs  # type: torch.Tensor
        self.transform = lambda x: x

    @classmethod
    def from_numpy(cls, inputs: np.array, outputs: np.array):
        """
        Alternative "constructor", straight from Numpy arrays
        """
        input_tensor = torch.from_numpy(inputs)
        output_tensor = torch.from_numpy(outputs)

        return cls(input_tensor, output_tensor)

    @classmethod
    def from_training_sample(cls, sample: TrainingSample, seq_len: int):
        """
        Alternative constructor: straight from TrainingSample, repeat
        """
        inputs = torch.tensor(
            np.repeat(np.array([sample.inputs]), seq_len, axis=0), dtype=settings.dtype
        )
        outputs = torch.tensor(
            np.repeat(np.array([sample.outputs]), seq_len, axis=0), dtype=settings.dtype
        )

        return cls(inputs, outputs)

    def append(self, inputs: torch.Tensor, outputs: torch.Tensor):
        """
        Concatenate another TrainingSet
        """
        assert inputs.shape[0] == outputs.shape[0], "Dimensions mismatch"

        self.inputs = torch.cat((self.inputs, inputs), 0).to(
            device=settings.device, dtype=settings.dtype
        )
        self.outputs = torch.cat((self.inputs, inputs), 0).to(
            device=settings.device, dtype=settings.dtype
        )

    def __getitem__(self, index):
        return self.transform(
            TrainingSample(
                inputs=self.inputs[index, :, :], outputs=self.outputs[index, :]
            )
        )

    def __len__(self):
        return self.inputs.shape[0]

    def set_transforms(self, transforms: List[Callable]):
        """
        Pass a list of transforms which are applied on a per sample fetch basis
        """
        self.transform = torchvision.transforms.Compose(transforms)


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
        mean = TrainingSample(
            torch.stack(mean_inputs).mean(dim=0), torch.stack(mean_outputs).mean(dim=0)
        )

        std = TrainingSample(
            torch.stack(std_inputs).mean(dim=0), torch.stack(std_outputs).mean(dim=0)
        )

        return mean, std

    def get_sequences(
        self, seq_len: int, type: torch.dtype = torch.float32
    ) -> TrainingSet:
        """
        Prepare sequences of a given length given the input data
        """

        inputs = []
        outputs = []

        for training_set in self.sets:
            a, b = self.generate_temporal_seq(
                training_set.inputs, training_set.outputs, seq_len
            )
            inputs.append(a)
            outputs.append(b)

        tensor_input = torch.cat(inputs).to(device=settings.device, dtype=type)
        tensor_output = torch.cat(outputs).to(device=settings.device, dtype=type)

        return TrainingSet(tensor_input, tensor_output)

    @staticmethod
    def generate_temporal_seq(
        tensor_input: torch.Tensor, tensor_output: torch.Tensor, seq_len: int
    ):
        """
        Generate all the subsequences over time,
        Useful for instance for training a temporal conv net
        """

        n_sequences = tensor_input.shape[0] - seq_len + 1

        input_seq = torch.transpose(
            torch.stack(
                [
                    tensor_input[start : start + seq_len, :]
                    for start in range(n_sequences)
                ],
                dim=0,
            ),
            1,
            2,
        )

        output_seq = tensor_output[: -seq_len + 1, :]

        return input_seq, output_seq

    def get_dataloaders(
        self,
        ratio: float,
        seq_len: int,
        batch_size: int,
        shuffle: bool,
        transforms: List[Callable],
        device: torch.device = torch.device("cpu"),
        dtype: torch.dtype = torch.float32,
    ) -> Tuple[DataLoader, DataLoader]:
        """
        Create two PyTorch DataLoaders out of this dataset, randomly splitting
        the data in between training and testing
        """
        # Create a consolidated dataset on the fly,
        # with the appropriate sequence cuts
        sequences = self.get_sequences(seq_len, type=dtype)
        sequences.set_transforms(transforms)

        # Split the dataset in train/test instances
        train_len = int(ratio * len(sequences))
        test_len = len(sequences) - train_len
        trainer, tester = random_split(sequences, [train_len, test_len])

        # Collate needs to enforce device and type somehow
        def collate(samples: List[TrainingSample]):
            return TrainingSample(
                inputs=torch.stack([t.inputs for t in samples])
                .squeeze()
                .to(device, dtype),
                outputs=torch.stack([t.outputs for t in samples])
                .squeeze()
                .to(device, dtype),
            )

        return (
            DataLoader(
                trainer,
                collate_fn=collate,
                batch_size=batch_size,
                shuffle=shuffle,
                drop_last=True,
            ),
            DataLoader(
                tester,
                collate_fn=collate,
                batch_size=batch_size,
                shuffle=shuffle,
                drop_last=True,
            ),
        )

    def get_sequential_sampler(
        self,
        seq_len: int,
        transforms: List[Callable],
        device: torch.device = torch.device("cpu"),
        dtype: torch.dtype = torch.float32,
    ) -> DataLoader:
        """
        Create two PyTorch DataLoaders out of this dataset, randomly splitting
        the data in between training and testing
        """
        # Create a consolidated dataset on the fly,
        # with the appropriate sequence cuts
        sequences = self.get_sequences(seq_len, type=dtype)
        sequences.set_transforms(transforms)

        # Collate needs to enforce device and type somehow
        def collate(samples: List[TrainingSample]):
            return TrainingSample(
                inputs=torch.stack([t.inputs for t in samples])
                .squeeze()
                .to(device, dtype),
                outputs=torch.stack([t.outputs for t in samples])
                .squeeze()
                .to(device, dtype),
            )

        return DataLoader(
            sequences, collate_fn=collate, batch_size=2048, drop_last=True
        )
