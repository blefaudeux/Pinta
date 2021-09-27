import torch
from pinta.rl.envs.base_env import BaseEnv
from pathlib import Path
from typing import Union, Optional, Tuple
from gym import spaces
import numpy as np
from pinta.data_processing.training_set import TrainingSample
from pinta.data_processing.transforms import Normalize


DEVICE = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")


class PintaEnv(BaseEnv):
    """Load a trained model, wrap it as a gym environement"""

    def __init__(
        self,
        model_path: str,
        target_twa: float,
        mean: TrainingSample,
        std: TrainingSample,
        max_iter: int = 1000,
        wind_speed: float = 10.0,
        max_rudder: float = 0.8,
        **_,
    ) -> None:
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
        self.wind_speed = wind_speed

        # Build a normalizer to go in and out DNN space
        normalizer = Normalize(mean, std)
        trajectory = []

        self.seed()
        self.reset()

    metadata = {"render.modes": ["human"]}

    @staticmethod
    def load_model(model_path: Union[str, Path], settings_path: Union[str, Path]) -> torch.nn.Module:
        import pinta.settings as settings
        from pinta.model.model_factory import model_factory

        # Load the saved pytorch nn
        training_settings = settings.load(settings_path)
        return model_factory(training_settings, model_path=model_path)

    def step(self, action: Tuple[float]):
        # apply the actions on the model, get the next state
        assert self.state is not None
        yaw, twa, speed = self.state

        # Feed the model with the current state and action, get the prediction and update the state
        # Recap:
        # TODO: make it dynamic given the settings ?
        # inputs: tws, twa_x, twa_y, helm
        # outputs: sog, twa_x, twa_y
        sample_inputs = torch.zeros((4,))
        sample_inputs[0] = self.wind_speed
        sample_inputs[1] = np.cos(twa)
        sample_inputs[2] = np.sin(twa)
        sample_inputs[3] = action

        TrainingSample(
            inputs=sample_inputs.unsqueeze(0),
            outputs=torch.zeros(1, 1),
        ).to(device=DEVICE)

        # TODO: pass in the trajectory over time to get a prediction

        # Compute the reward
        reward = -1

        # A Gym env returns 4 objects:
        # onservation, reward, done and optional info
        return self.state, reward, self.iter > self.max_iter, {}

    def reset(self):
        # reload the model
        self.state = np.array([0.0, 0.0, 0.0])

        self.model = self.load_model(self._model_path, self._settings_path).to(DEVICE)

        self.iter = 0
        self.steps_beyond_done = None
        return self.state
