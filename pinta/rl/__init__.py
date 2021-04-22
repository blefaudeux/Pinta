from gym.envs.registration import register
from pinta.rl.envs import PintaEnv

register(
    id='Pinta',
    entry_point='pinta.rl.envs:PintaEnv',
)