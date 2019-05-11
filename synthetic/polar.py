
import numpy as np
from collections import namedtuple

SpeedPolarPoint = namedtuple(
    "SpeedPolarPoint", ["wind_angle",  "wind_speed", "rudder_angle", "boat_speed"])

# Generate data points across the whole range


def generate_data(engine, wind_range, wind_step, angular_step):
    speed = []

    for w in range(wind_range[0], wind_range[1], wind_step):
        for a in range(0, np.pi, angular_step):
            pt = SpeedPolarPoint()
            pt.wind_angle = a
            pt.wind_speed = w
            pt.rudder_angle = 0.
            pt.boat_speed = engine.predict(w, a, 0.)
            speed.append(pt)

    return speed
