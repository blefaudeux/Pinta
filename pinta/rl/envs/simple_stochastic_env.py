import gym
from gym import spaces
from gym.utils import seeding
import math
from typing import Tuple
import numpy as np
from gym.envs.classic_control import rendering


class SimpleStochasticEnv(gym.Env):
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
        self, white_noise: float, slow_moving_noise: float, inertia: float, target_twa: float, max_iter: int = 1000
    ):
        self.white_noise = white_noise
        self.slow_moving_noise = slow_moving_noise
        self.inertia = inertia
        self.target_twa = np.array([target_twa])

        self.state = None
        self.max_iter = max_iter
        self.steps_beyond_done = None
        self.viewer = None

        # Action space is the rudder
        self.action_space = spaces.Box(low=np.array([-0.5]), high=np.array([0.5]), shape=(1,), dtype=np.float32)

        # Observation space is the current yaw, boat speed and TWA
        self.observation_space = spaces.Box(
            low=np.array([-np.pi, -np.pi, 0.0]), high=np.array([np.pi, np.pi, 3.0]), shape=(3,), dtype=np.float32
        )

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

        yaw, twa, speed = self.state

        # Given the rudder action and some noise, compute the new state
        # - approximate rudder action
        yaw_diff = action * speed
        yaw += yaw_diff
        twa += yaw_diff + self.np_random.uniform(low=-self.white_noise, high=self.white_noise)
        speed = np.array([self.inertia * speed + (1.0 - self.inertia) * self._speed(twa)])

        # Reward is just cos(twa, target_twa)
        reward = (1.0 + np.cos(twa, self.target_twa)) / 2.0

        # Update the state, and good to go
        self.state = np.concatenate([yaw, twa, speed])
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
        self.state[1] = self.target_twa + self.np_random.uniform(low=-0.1, high=0.1, size=(1,))
        self.state[2] = self._speed(self.state[1])
        self.iter = 0
        self.steps_beyond_done = None
        return self.state

    def render(self, mode="human"):
        screen_width = 600
        screen_height = 400

        # Draw the boat and the current wind
        if self.viewer is None:
            boat_len = 80
            boat_width = 5
            wind_len = 20
            wind_width = 2

            metrics_len = 100
            metrics_width = 5

            # initial setup
            # - create the boat geometry
            self.viewer = rendering.Viewer(screen_width, screen_height)
            l, r, t, b = -boat_width / 2, boat_width / 2, boat_len / 2, -boat_len / 2
            boat = rendering.FilledPolygon([(l, b), (l, t), (r, t), (r, b)])

            # - create the initial boat transform, then commit to the viewer
            self.trans_boat = rendering.Transform(translation=(screen_width // 2, screen_height // 2))
            boat.add_attr(self.trans_boat)
            self.viewer.add_geom(boat)

            # - now add the wind geometry
            l, r, t, b = -wind_width / 2, wind_width / 2, wind_len / 2, -wind_len / 2
            wind = rendering.FilledPolygon([(l, b), (l, t), (r, t), (r, b)])
            wind.set_color(0.8, 0.6, 0.4)

            # - and the corresponding boat transform, then commit to the viewer
            axleoffset = boat_len / 4.0
            self.trans_wind = rendering.Transform(translation=(screen_width // 2, screen_height // 2 + axleoffset))
            wind.add_attr(self.trans_wind)
            self.viewer.add_geom(wind)

            # - add some metrics, current twa and target
            l, r, t, b = -metrics_width / 2, metrics_width / 2, metrics_len / 2, -metrics_len / 2
            metrics_twa_target = rendering.FilledPolygon([(l, b), (l, t), (r, t), (r, b)])
            self.trans_metrics_twa_target = rendering.Transform(translation=(screen_width - 20, 0))
            metrics_twa_target.add_attr(self.trans_metrics_twa_target)
            metrics_twa_target.set_color(0.0, 1.0, 0.0)
            self.viewer.add_geom(metrics_twa_target)

            metrics_twa = rendering.FilledPolygon([(l, b), (l, t), (r, t), (r, b)])
            self.trans_metrics_twa = rendering.Transform(translation=(screen_width - 40, 0))
            metrics_twa.add_attr(self.trans_metrics_twa)
            self.scale_metrics_twa = rendering.Transform(scale=(1.0, 1.0))
            metrics_twa.add_attr(self.scale_metrics_twa)
            metrics_twa.set_color(1.0, 0.0, 0.0)
            self.viewer.add_geom(metrics_twa)

            metrics_speed = rendering.FilledPolygon([(l, b), (l, t), (r, t), (r, b)])
            self.trans_metrics_speed = rendering.Transform(translation=(40, 0))
            metrics_speed.add_attr(self.trans_metrics_speed)
            self.scale_metrics_speed = rendering.Transform(scale=(1.0, 1.0))
            metrics_speed.add_attr(self.scale_metrics_speed)
            metrics_speed.set_color(0.0, 0.0, 1.0)
            self.viewer.add_geom(metrics_speed)

        if self.state is None:
            return None

        # Move the boat and the wind
        yaw, twa, speed = self.state
        self.trans_wind.set_translation(0, screen_height // 2)
        self.trans_wind.set_rotation(twa)
        self.trans_boat.set_rotation(yaw)
        self.scale_metrics_twa.set_scale(1, 1 - (twa - self.target_twa) / self.target_twa)
        self.scale_metrics_speed.set_scale(1, speed)

        return self.viewer.render(return_rgb_array=mode == "rgb_array")

    def close(self):
        if self.viewer:
            self.viewer.close()
            self.viewer = None

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]
