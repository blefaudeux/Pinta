from gym.core import Env
import numpy as np
from behaviour import NN
import collections


# To be defined:
# - how to test mutiple observation points over time ? Typical model
# seems to involve a purely sequential state.
# Use a random seed to switch state ?

BUFFER_LENGTH = 20


class BoatNN(Env):
  """
  An OpenAI-compatible environment simulating a boat behaviour.

  Main idea is to reuse the NN trained otherwise,
  which only solves part of the problem :
  - Computing the reward on a reasonable basis is still to be determined
  - We would need a way to make sure that every situation is properly optimized for,
  not just from a given starting point
  """

  # Inherited attributes
  action_space = 1
  observation_space = 1 + 3  # Speed + Attitude ?
  reward_range = (-np.inf, np.inf)

  def __init__(self):
    self.prev_obs_state = np.zeros(observation_space)
    self.prev_act_state = np.zeros(action_space)
    self.prev_reward = 0.
    self.state_buffer = collections.deque(BUFFER_LENGTH)
    # Default NN, the proper model is set in the configure() step
    self.nn = NN()

  # Inherited methods
  def _step(self, action):
    """Run one timestep of the environment's dynamics. When end of
    episode is reached, you are responsible for calling `reset()`
    to reset this environment's state.

    Accepts an action and returns a tuple (observation, reward, done, info).

    Args:
        action (object): an action provided by the environment

    Returns:
        observation (object): agent's observation of the current environment
        reward (float) : amount of reward returned after previous action
        done (boolean): whether the episode has ended, in which case further step() calls will return undefined results
        info (dict): contains auxiliary diagnostic information (helpful for debugging, and sometimes learning)
    """

    # TODO: Compute a reward, diagnostic info, etc.. not quite there yet
    self.prev_obs_state = self.nn.predict(action)
    self.prev_act_state = action
    return self.prev_obs_state

  def _reset(self):
    """Resets the state of the environment and returns an initial
    observation. Will call 'configure()' if not already called.

    Returns: observation (object): the initial observation of the
        space. (Initial reward is assumed to be 0.)
    """
    raise NotImplementedError

  def _render(self, mode='human', close=False):
    if close:
      return
    print("No implemented render for now, could be an option to \
          visualize the boat state live")

  def _seed(self, seed=None):
    """Sets the seed for this env's random number generator(s).

    Note:
        Some environments use multiple pseudorandom number generators.
        We want to capture all such seeds used in order to ensure that
        there aren't accidental correlations between multiple generators.

    Returns:
        list<bigint>: Returns the list of seeds used in this env's random
          number generators. The first value in the list should be the
          "main" seed, or the value which a reproducer should pass to
          'seed'. Often, the main seed equals the provided 'seed', but
          this won't be true if seed=None, for example.
    """
    return []

  def _close(self):
    pass

  def _configure(self, nn_to_load):
    """Provides runtime configuration to the environment.
    Choose the trained NN to load for behavioural simulation
    """
    self.nn.load(nn_to_load)

  def _compute_reward(self):
    # TODO: consider the buffered states,
    # - average route should be respected
    # - speed should be maximized

    raise NotImplementedError
