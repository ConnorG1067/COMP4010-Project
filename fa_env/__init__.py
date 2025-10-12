from gymnasium.envs.registration import register

register(
    id="fa_env/GridWorld-v0",
    entry_point="fa_env.envs:GridWorldEnv",
)
