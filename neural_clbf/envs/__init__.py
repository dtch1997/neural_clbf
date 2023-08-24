from gym import register 

register("CAF-InvertedPendulum-v2", entry_point="neural_clbf.envs.env:OfflineInvertedPendulumEnv", max_episode_steps=100)
