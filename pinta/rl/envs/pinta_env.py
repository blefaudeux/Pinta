import torch
from pinta.rl.envs.base_env import BaseEnv
from pathlib import Path
from typing import Union, Optional, Tuple
from gym import spaces
import numpy as np


def load_model(model_path: Union[str, Path], settings_path: Union[str, Path]) -> torch.nn.Module:
    import pinta.settings as settings
    from pinta.model.model_factory import model_factory

    # Load the saved pytorch nn
    training_settings = settings.load(settings_path)
    return model_factory(training_settings, model_path=model_path)


class PintaEnv(BaseEnv):
    """Load a trained model, wrap it as a gym environement"""

    def __init__(self, model_path: str, target_twa: float, max_iter: int = 1000, max_rudder: float = 0.8, **_) -> None:
        self._model_path = Path(model_path)
        self._settings_path = Path("settings.json")

        self.state = None
        self.max_iter = max_iter
        self.target_twa = target_twa
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

    metadata = {"render.modes": ["human"]}

    def step(self, action: Tuple[float]):
        # apply the actions on the model, get the next state
        assert self.state is not None

        # Feed the model with the current state and action, get the prediction and update the state
        # TODO

        # Compute the reward
        reward = -1

        # A Gym env returns 4 objects:
        # onservation, reward, done and optional info
        return self.state, reward, self.iter > self.max_iter, {}

    def reset(self):
        # reload the model
        self.state = np.array([0.0, 0.0, 0.0])

        self.model = load_model(self._model_path, self._settings_path)

        self.iter = 0
        self.steps_beyond_done = None
        return self.state
