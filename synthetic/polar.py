from collections import namedtuple

import numpy as np

from data_processing.training_set import TrainingSample

SpeedPolarPoint = namedtuple(
    "SpeedPolarPoint", ["wind_angle", "wind_speed", "rudder_angle", "boat_speed"]
)


def generate(engine, wind_range, wind_step, angular_step, seq_len):
    speed = []

    # Build an artificial dataloader
    # FIXME: Ben
    for w in range(wind_range[0], wind_range[1], wind_step):
        for a in np.arange(0.0, np.pi, angular_step):
            _ = TrainingSample(
                inputs=np.array([w, np.cos(a), np.sin(a), 0.0]), outputs=np.array([0.0])
            )

    # FW pass in the DNN and save
    # FIXME: Ben. Broken interface. Would need unit testing

    # pred = engine.predict(sample_input, seq_len)
    # pred = pred.detach().cpu().numpy()

    # speed.append(
    #     SpeedPolarPoint(wind_angle=a, wind_speed=w, rudder_angle=0.0, boat_speed=pred)
    # )

    return speed
