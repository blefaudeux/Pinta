from gym.envs.registration import register

register(
    id="SimpleStochasticEnv-v0",
    entry_point="pinta.rl.envs:SimpleStochasticEnv",
)


register(
    id="PintaEnv-v0",
    entry_point="pinta.rl.envs:PintaEnv",
)
