from gym import spaces
from gym.utils import seeding
import math
from typing import Tuple, Optional
import numpy as np
from pinta.rl.envs.pinta_env import BaseEnv


class SimpleStochasticEnv(BaseEnv):
    """
    Description:
        The boat uses a simple law to estimate its theoretical speed with respect to the true wind angle,
            `s(twa) = sin(twa) + sqrt(twa)`

        To simulate inertia, a moving average is used

        Some noise is added to its attitude over time, to simulate the action of waves for instance,
        or a noisy wind force and direction


    Observation:
        Type: Box(2)
        Num     Observation     Min    Max
        0       Boat Velocity   0      3.
        1       Boat TWA        -Pi    Pi

     Actions:
        Type: Continuous(1)
        Num   Action
        0     Rudder angle

    Reward:
        Reward is 1 if the boat is precisely on the TWA target, 0 if opposed.

    Starting State:
        The boat has a starting position close to the target + random gaussian noise

    """

    metadata = {"render.modes": ["human"]}

    def __init__(
        self,
        white_noise: float,
        slow_moving_noise: float,
        inertia: float,
        target_twa: float,
        max_rudder: float = 0.8,
        max_iter: int = 1000,
    ):
        self.white_noise = white_noise
        self.slow_moving_noise = slow_moving_noise
        self.inertia = inertia
        self.target_twa = np.array([target_twa])

        self.state = None
        self.max_iter = max_iter
        self.steps_beyond_done: Optional[int] = None
        self.viewer = None

        # Action space is the rudder
        self.action_space = spaces.Box(
            low=np.array([-max_rudder]),
            high=np.array([max_rudder]),
            shape=(1,),
            dtype=np.float32,
        )

        # Observation space is the current yaw, boat speed and TWA
        self.observation_space = spaces.Box(
            low=np.array([-np.pi, -np.pi, 0.0]),
            high=np.array([np.pi, np.pi, 3.0]),
            shape=(3,),
            dtype=np.float32,
        )
        self.rudder = 0.0

        self.seed()
        self.reset()

    @staticmethod
    def _speed(twa: float):
        twa_ = math.fabs(twa)
        return math.sin(twa_) + math.sqrt(twa_)

    def step(self, action: Tuple[float]):
        if not self.action_space.contains(action):
            assert self.steps_beyond_done is None
            self.steps_beyond_done = 0
            return self.state, 0.0, True, {}

        assert self.state is not None

        self.rudder = action
        yaw, twa, speed = self.state

        # Given the rudder action and some noise, compute the new state
        # - approximate rudder action
        yaw_diff = 5e-2 * self.rudder * speed
        yaw += yaw_diff
        twa += yaw_diff + self.np_random.normal(loc=0, scale=self.white_noise)

        # Speed is based on inertia + TWA
        speed = self.inertia * speed + (1.0 - self.inertia) * np.array([self._speed(twa)])

        # Changing direction costs some speed
        speed *= 1.0 - abs(self.rudder ** 2)

        # Reward needs to take alignment and wind side into account
        reward = np.cos(twa - self.target_twa)

        # Update the state, and good to go
        self.state = np.array([yaw[0], twa[0], speed[0]])
        self.iter += 1

        # A Gym env returns 4 objects:
        # onservation, reward, done and optional info
        return self.state, reward, self.iter > self.max_iter, {}

    def reset(self):
        # State is:
        # - yaw
        # - true wind angle
        # - boat speed

        self.state = np.array([0.0, 0.0, 0.0])

        # Randomly place the boat with respect to the wind
        self.state[1] = self.np_random.uniform(low=-3.14, high=3.14, size=(1,))
        self.state[2] = self._speed(self.state[1])
        self.iter = 0
        self.steps_beyond_done = None
        return self.state

    def close(self):
        if self.viewer:
            self.viewer.close()
            self.viewer = None

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]
