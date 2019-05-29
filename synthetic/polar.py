
import numpy as np
from collections import namedtuple

SpeedPolarPoint = namedtuple(
    "SpeedPolarPoint", ["wind_angle", "wind_speed", "rudder_angle", "boat_speed"])


def generate(engine, wind_range, wind_step, angular_step):
    speed = []

    for w in range(wind_range[0], wind_range[1], wind_step):
        for a in np.arange(0., np.pi, angular_step):
            pt = SpeedPolarPoint(wind_angle=a,
                                 wind_speed=w,
                                 rudder_angle=0.,
                                 boat_speed=engine.predict(w, a, 0.))
            speed.append(pt)

    return speed
