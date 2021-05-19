from __future__ import annotations

# Our lightweight base data structure..
# specialize inputs/outputs, makes it readable down the line
from collections import namedtuple
from typing import Callable, List, Tuple, Optional
from pinta.settings import Settings

import numpy as np
import torch
import torchvision
from torch.utils.data import DataLoader, Dataset, random_split

TrainingSample_base = namedtuple("TrainingSample_base", ["inputs", "outputs"])


class TrainingSample(TrainingSample_base):
    """
    Holds one or many training samples. Make sure that the dimensions
    and matching in between inputs and outputs are always respected
    """

    def to(self, device: torch.device = None) -> TrainingSample:
        return TrainingSample(inputs=self.inputs.to(device), outputs=self.outputs.to(device))

    def __str__(self):
        inputs_str = "".join(["{:.2f} ".format(i) for i in self.inputs.cpu().tolist()])
        outputs_str = "".join(["{:.2f} ".format(i) for i in self.outputs.cpu().tolist()])

        return f"inputs: {inputs_str}\noutputs: {outputs_str}\n"


class TrainingSet(Dataset):
    """
    Holds the training or testing data, with some helper functions.
    This keeps all the time coherent data in one package.

    .. warning: Time incoherent data should not be stored in a TrainingSet,
        sequences sampled out of it could overlap with boundaries and become nonsensical

    .. note: Expected shape is [Time sample x Channels]
    """

    def __init__(self, inputs: torch.Tensor, outputs: torch.Tensor):
        assert inputs.shape[0] == outputs.shape[0], "Dimensions mismatch"

        self.inputs = inputs  # type: torch.Tensor
        self.outputs = outputs  # type: torch.Tensor
        self.transform = lambda x: x
        self.inputs_seq_view: Optional[torch.Tensor] = None
        self.outputs_seq_view: Optional[torch.Tensor] = None

    @classmethod
    def from_numpy(cls, inputs: np.array, outputs: np.array):  # type: ignore
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
        Expected size is [Time sequence x Channels]
        """
        return cls(sample.inputs.repeat(seq_len, 1), sample.outputs.repeat(seq_len, 1))

    def append(self, inputs: torch.Tensor, outputs: torch.Tensor):
        """
        Concatenate another TrainingSet
        """
        assert inputs.shape[1] == outputs.shape[1], "Dimensions mismatch"

        self.inputs = torch.cat((self.inputs, inputs), 0)
        self.outputs = torch.cat((self.outputs, outputs), 0)

    def __getitem__(self, index) -> TrainingSample:
        return self.transform(TrainingSample(inputs=self.inputs[index, :], outputs=self.outputs[index, :]))

    def get_sequence(self, index: int, seq_length: int) -> TrainingSample:
        index = min(index, self.inputs.shape[0] - seq_length)  # Handle the sequence length overflowing the dataset

        if self.inputs_seq_view is None or self.outputs_seq_view is None:
            self.inputs_seq_view = self.inputs.unfold(0, seq_length, 1)
            self.outputs_seq_view = self.outputs.unfold(0, seq_length, 1)

        return self.transform(TrainingSample(inputs=self.inputs_seq_view[index], outputs=self.outputs_seq_view[index]))

    def __len__(self):
        return self.inputs.shape[0]

    def set_transforms(self, transforms: List[Callable]):
        """
        Pass a list of transforms which are applied on a per sample fetch basis
        """
        self.transform = torchvision.transforms.Compose(transforms)


class TrainingSetBundle(Dataset):
    """
    Hold a list of training sets, with some helper functions.

    This allows us to maintain a bigger pool of data
    without any time coherency / time continuity constraints.

    All the data can be used for training, but we can enforce
    time-continuous streams where needed.
    """

    def __init__(self, training_sets: List[TrainingSet]):
        self.sets = training_sets
        self.transform = lambda x: x
        # Build the index map, used to redirect __get_item__ to the proper
        # dataset
        self.index_map = [0]
        for s in self.sets:
            self.index_map.append(len(s) + self.index_map[-1])
        self.seq_length = -1

    def __len__(self):
        return sum(map(lambda x: len(x), self.sets))

    def set_transforms(self, transforms: List[Callable]):
        """
        Pass a list of transforms which are applied on a per sample fetch basis
        """
        self.transform = torchvision.transforms.Compose(transforms)

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
        mean = TrainingSample(torch.stack(mean_inputs).mean(dim=0), torch.stack(mean_outputs).mean(dim=0))
        std = TrainingSample(torch.stack(std_inputs).mean(dim=0), torch.stack(std_outputs).mean(dim=0))

        return mean, std

    def __getitem__(self, index):
        # Find the relevant TrainingSet.
        match = next(i for i, x in enumerate(map(lambda x: x > index, self.index_map)) if x) - 1
        local_index = index - self.index_map[match]

        # Use the TrainingSet to fetch the appropriate sequence, transform and return
        sequence = self.sets[match].get_sequence(local_index, self.seq_length)
        return self.transform(sequence)

    def get_training_set(self, seq_len: int) -> Tuple[TrainingSet, List[int]]:
        """
        Generate a single TrainingSet from a bundle.

        .. warning: this can take a sizeable amount of memory, all the sequences will be materialized.
        Use .get_item() if you want to stream the dataset
        """

        inputs = []
        outputs = []

        for training_set in self.sets:
            # Generate memory views (no copies) with all the possible sequences
            inputs.append(training_set.inputs.unfold(0, seq_len, 1))
            outputs.append(training_set.outputs.unfold(0, seq_len, 1))

        return (
            TrainingSet(torch.cat(inputs), torch.cat(outputs)),
            [len(sequence) for sequence in inputs],
        )

    def get_dataloaders(
        self,
        params: Settings,
        transforms: List[Callable],
    ) -> Tuple[DataLoader, DataLoader]:
        """
        Create two PyTorch DataLoaders out of this dataset, randomly splitting
        the data in between training and testing
        """

        # Split the dataset in train/test instances
        self.seq_length = params.trunk.seq_length
        self.set_transforms(transforms)
        train_len = int(params.data.training_ratio * len(self))
        test_len = len(self) - train_len
        trainer, tester = random_split(self, [train_len, test_len])

        def collate(samples: List[TrainingSample]):
            """
            Dimensions are [Batch x Channels x TimeSequence].
            """
            return TrainingSample(
                inputs=torch.stack([t.inputs for t in samples]),
                outputs=torch.stack([t.outputs for t in samples]),
            )

        return (
            DataLoader(
                trainer,
                collate_fn=collate,
                batch_size=params.data.train_batch_size,
                shuffle=params.data.shuffle,
                drop_last=True,
                num_workers=params.data.train_workers,
            ),
            DataLoader(
                tester,
                collate_fn=collate,
                batch_size=params.data.test_batch_size,
                shuffle=params.data.shuffle,
                drop_last=True,
                num_workers=params.data.test_workers,
            ),
        )

    def get_sequential_dataloader(
        self, seq_len: int, transforms: List[Callable], batch_size: int = 8000
    ) -> Tuple[DataLoader, List[int]]:
        """
        Create a sequential PyTorch DataLoader out of this dataset,
        along with the indices which make it possible to align the sequences
        """

        training_set, split_indices = self.get_training_set(seq_len)
        training_set.set_transforms(transforms)

        def collate(samples: List[TrainingSample]):
            return TrainingSample(
                inputs=torch.stack([t.inputs for t in samples]).permute(0, 2, 1),
                outputs=torch.stack([t.outputs for t in samples]).permute(0, 2, 1),
            )

        return (
            DataLoader(
                training_set,
                collate_fn=collate,
                batch_size=batch_size,
            ),
            split_indices,
        )
