import gym
from gym.envs.classic_control import rendering
from gym.utils import seeding


class BaseEnv(gym.Env):
    def __init__(self) -> None:
        ...

    @staticmethod
    def _get_polygon(width, height):
        l, r, t, b = -width / 2, width / 2, height / 2, -height / 2
        return [(l, b), (l, t), (r, t), (r, b)]

    def _setup_viewer(self):
        screen_width = 800
        screen_height = 800

        boat_len = 160
        boat_width = 10

        rudder_len = 20
        rudder_width = 6

        wind_len = 60
        wind_width = 10

        metrics_len = 100
        metrics_width = 5

        # initial setup
        # - create the boat geometry
        self.viewer = rendering.Viewer(screen_width, screen_height)
        boat = rendering.FilledPolygon(self._get_polygon(boat_width, boat_len))

        # - create the initial boat transform, then commit to the viewer
        self.trans_boat = rendering.Transform(translation=(screen_width // 2, screen_height // 2))
        boat.add_attr(self.trans_boat)
        self.viewer.add_geom(boat)

        # - now add the wind geometry
        wind = rendering.FilledPolygon(self._get_polygon(wind_width, wind_len))
        wind.set_color(0.8, 0.6, 0.4)

        # - and the corresponding boat transform, then commit to the viewer
        wind_offset = 0.8 * screen_height
        self.trans_wind = rendering.Transform(translation=(screen_width // 2, wind_offset))
        wind.add_attr(self.trans_wind)
        self.viewer.add_geom(wind)

        # - now add the rudder geometry
        rudder = rendering.FilledPolygon(self._get_polygon(rudder_width, rudder_len))
        rudder.set_color(0.3, 0.3, 0.3)

        # - and the corresponding boat transform, then commit to the viewer
        rudder_offset = boat_len / 2.0
        self.trans_rudder = rendering.Transform(translation=(0, -rudder_offset))
        rudder.add_attr(self.trans_rudder)
        rudder.add_attr(self.trans_boat)
        self.viewer.add_geom(rudder)

        # - add some metrics, current twa and target
        metrics_twa_target = rendering.FilledPolygon(self._get_polygon(metrics_width, metrics_len))
        self.trans_metrics_twa_target = rendering.Transform(translation=(screen_width - 20, 0))
        metrics_twa_target.add_attr(self.trans_metrics_twa_target)
        metrics_twa_target.set_color(0.0, 1.0, 0.0)
        self.viewer.add_geom(metrics_twa_target)

        metrics_twa = rendering.FilledPolygon(self._get_polygon(metrics_width, metrics_len))
        self.trans_metrics_twa = rendering.Transform(translation=(screen_width - 40, 0))
        metrics_twa.add_attr(self.trans_metrics_twa)
        self.scale_metrics_twa = rendering.Transform(scale=(1.0, 1.0))
        metrics_twa.add_attr(self.scale_metrics_twa)
        metrics_twa.set_color(1.0, 0.0, 0.0)
        self.viewer.add_geom(metrics_twa)

        # -  add speed metrics
        metrics_speed = rendering.FilledPolygon(self._get_polygon(metrics_width, metrics_len))
        self.trans_metrics_speed = rendering.Transform(translation=(40, 0))
        metrics_speed.add_attr(self.trans_metrics_speed)

        self.scale_metrics_speed = rendering.Transform(scale=(1.0, 1.0))
        metrics_speed.add_attr(self.scale_metrics_speed)

        metrics_speed.set_color(0.0, 0.0, 1.0)
        self.viewer.add_geom(metrics_speed)

    def render(self, mode="human"):
        if self.state is None:
            return None

        # Draw the boat and the current wind
        if self.viewer is None:
            self._setup_viewer()

        # Move the boat and the wind
        yaw, twa, speed = self.state
        self.trans_wind.set_rotation(yaw - twa)
        self.trans_boat.set_rotation(yaw)
        self.trans_rudder.set_rotation(-self.rudder)
        self.scale_metrics_twa.set_scale(1, 1 - (twa - self.target_twa) / self.target_twa)
        self.scale_metrics_speed.set_scale(1, speed)

        return self.viewer.render(return_rgb_array=mode == "rgb_array")

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]
