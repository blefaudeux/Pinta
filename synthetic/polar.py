from collections import namedtuple
from typing import List

import numpy as np
import torch
from torch.utils.data import DataLoader

from data_processing.training_set import (
    TrainingSample,
    TrainingSet,
    TrainingSetBundle,
)
from data_processing.transforms import Normalize
from settings import device

SpeedPolarPoint = namedtuple("SpeedPolarPoint", ["twa", "tws", "helm", "boat_speed"])


def generate(
    engine,
    wind_range: List[int],
    wind_step: int,
    angular_step: float,
    seq_len: int,
    mean: TrainingSample,
    std: TrainingSample,
):
    # Build an artificial TrainingSet
    # - one single measurement point per polar coord
    # (repeat the same inputs for the whole sequence)
    datasets = []

    normalizer = Normalize(mean.to(device, torch.float), std.to(device, torch.float),)

    for w in range(wind_range[0], wind_range[1], wind_step):
        for a in np.arange(0.0, np.pi, angular_step):
            # Generate the sample we want for this point in the polar plot
            # Expected size for transform is [Time x Channels],
            # so we unsqueeze front
            sample = TrainingSample(
                inputs=torch.tensor([w, np.cos(a), np.sin(a), 0.0], device=device).unsqueeze(0),
                outputs=torch.zeros(1, 1).to(device=device),
            )

            # Normalize to get to the range the model has learnt
            datasets.append(TrainingSet.from_training_sample(normalizer(sample), seq_len))

    # - build a collection of unrelated points from this,
    # a TrainingSetBUndle
    dataset_bundle = TrainingSetBundle(datasets)

    # - get a dataloader from the ad-hoc dataset
    dataloader = DataLoader(dataset_bundle.get_training_set(seq_len)[0], batch_size=100, shuffle=False)

    # FW pass in the DNN.
    # Passing in the mean and std allows for de-normalization on the way out
    pred = engine.predict(dataloader, mean.outputs.to(device=device), std.outputs.to(device=device))
    pred = pred.detach().cpu().numpy()

    # Unpack, generate the appropriate list of points for plotting
    i_pred = 0
    speeds = []
    for w in range(wind_range[0], wind_range[1], wind_step):
        for a in np.arange(0.0, np.pi, angular_step):
            speeds.append(SpeedPolarPoint(twa=a, tws=w, helm=0.0, boat_speed=pred[i_pred],))
            i_pred += 1

    return speeds
