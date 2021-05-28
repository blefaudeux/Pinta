import gym
import torch
import pinta.settings as settings
from pinta.model.model_factory import model_factory


def load_model(model_path: str, settings_path: str) -> torch.nn.Module:
    # Load the saved pytorch nn
    training_settings = settings.load(settings_path)
    return model_factory(training_settings, model_path=model_path)


class PintaEnv(gym.Env):
    """ Load a trained model, wrap it as a gym environement """

    def __init__(self) -> None:
        self.model = load_model(".model.pt", ".settings.json")

    metadata = {"render.modes": ["human"]}

    def step(self, action):
        # apply the actions on the model, get the next state
        ...

    def reset(self):
        # reload the model
        ...

    def render(self, mode="human", close=False):
        # TODO: quick pyglet viewer or something related ?
        ...