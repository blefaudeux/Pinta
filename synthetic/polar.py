from collections import namedtuple

import numpy as np
import torch
from torch.utils.data import DataLoader

from data_processing.training_set import TrainingSet

SpeedPolarPoint = namedtuple(
    "SpeedPolarPoint", ["wind_angle", "wind_speed", "rudder_angle", "boat_speed"]
)


def generate(engine, wind_range, wind_step, angular_step, seq_len):
    # Build an artificial TrainingSet
    # - one single measurement point per polar coord
    # (repeat the same inputs for the whole sequence)
    inputs = []
    outputs = []

    for w in range(wind_range[0], wind_range[1], wind_step):
        for a in np.arange(0.0, np.pi, angular_step):
            inputs.append(
                torch.tensor([w, np.cos(a), np.sin(a), 0.0]).repeat(seq_len, 1)
            )
            outputs.append(torch.zeros(1, seq_len))

    # - build a collection of unrelated points from this,
    # a TrainingSetBUndle
    dataset = TrainingSet(torch.stack(inputs).permute(0, 2, 1), torch.stack(outputs))

    # - get a dataloader from the ad-hoc dataset
    dataloader = DataLoader(dataset, batch_size=100, shuffle=False)

    # FW pass in the DNN and save
    pred = engine.predict(dataloader, seq_len)
    pred = pred.detach().cpu().numpy()

    # TODO:
    # - unpack the results so that they fit the expected format for plotting
    # - handle de-normalization

    # speed.append(
    #     SpeedPolarPoint(wind_angle=a, wind_speed=w, rudder_angle=0.0, boat_speed=pred)
    # )

    # return speed
