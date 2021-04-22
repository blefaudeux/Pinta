import torch
import pinta.settings  as settings
from pinta.model.model_factory import model_factory
import gym
from gym import error, spaces, utils
from gym.utils import seeding


""" Load a trained model, wrap it as a gym environement """

def load_model(model_path: str, settings_path: str) -> torch.nn.Module:
    # Load the saved pytorch nn
    training_settings = settings.load(settings_path)
    return model_factory(training_settings, model_path=model_path)


class PintaEnv(gym.Env):
    #TODO: Ben
    def __init__(self) -> None:
        pass

    metadata = {'render.modes': ['human']}


    def step(self, action):
        ...

    def reset(self):
        ...

    def render(self, mode='human', close=False):
        ...
