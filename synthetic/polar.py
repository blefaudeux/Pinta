
import numpy as np
from collections import namedtuple
from settings import TrainingSet

SpeedPolarPoint = namedtuple(
    "SpeedPolarPoint", ["wind_angle", "wind_speed", "rudder_angle", "boat_speed"])


def generate(engine, wind_range, wind_step, angular_step, seq_len):
    speed = []

    for w in range(wind_range[0], wind_range[1], wind_step):
        for a in np.arange(0., np.pi, angular_step):
            sample_input = TrainingSet(
                np.array([w, np.cos(a), np.sin(a), 0.]), np.array([0.]))

            # FW pass in the DNN and save
            pt = SpeedPolarPoint(wind_angle=a,
                                 wind_speed=w,
                                 rudder_angle=0.,
                                 boat_speed=engine.predict(sample_input, seq_len))
            speed.append(pt)

    return speed
