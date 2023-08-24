import gym
import numpy as np
import torch
from d4rl import offline_env
from neural_clbf.systems import ControlAffineSystem
from neural_clbf.systems.inverted_pendulum import InvertedPendulum

class ControlAffineSystemEnv(gym.Env):
    def __init__(self, system: ControlAffineSystem):
        self.system = system
        self.action_space = gym.spaces.Box(
            low=system.control_limits[1].cpu().numpy(), 
            high=system.control_limits[0].cpu().numpy()
        )
        self.observation_space = gym.spaces.Box(
            low= system.state_limits[1].cpu().numpy(), 
            high= system.state_limits[0].cpu().numpy()
        )
        self.state_th = self.system.sample_safe(1)

    def reset(self):
        self.state_th = self.system.sample_safe(1)
        return self.state_th.squeeze(0).cpu().numpy()

    def step(self, action: np.ndarray):
        # action = np.clip(action, self.action_space.low, self.action_space.high)
        action = torch.from_numpy(action).float().unsqueeze(0)
        next_state_th = self.system.zero_order_hold(
            self.state_th, action, self.system.dt
        )
        done = self.system.safe_mask(next_state_th).cpu().numpy()
        done = bool(done)

        # Sparse goal-reaching reward
        if self.system.goal_mask(next_state_th):
            reward = 1.0
        else:
            reward = 0.0
        return self.state_th.cpu().squeeze(0).numpy(), reward, done, {}

class OfflineControlAffineSystemEnv(ControlAffineSystemEnv, offline_env.OfflineEnv):
    def __init__(self, system: ControlAffineSystem):
        ControlAffineSystem.__init__(self, system)
    
class OfflineInvertedPendulumEnv(OfflineControlAffineSystemEnv):
    def __init__(self, dataset_path: str = 'neural_clbf/envs/data/inverted_pendulum_random.pkl'):
        nominal_params = {"m": 1.0, "L": 1.0, "b": 0.01}
        super(OfflineControlAffineSystemEnv, self).__init__(InvertedPendulum(nominal_params=nominal_params)) 
        self.dataset_path = dataset_path

    def get_dataset(self):
        import pickle
        with open(self.dataset_path, "rb") as f:
            data = pickle.load(f)
        return data

if __name__ == "__main__":

    import d4rl
    env = OfflineInvertedPendulumEnv()
    env = gym.wrappers.TimeLimit(env, max_episode_steps=100)
    # Test dataset
    dataset = d4rl.qlearning_dataset(env)

    # Sanity check env
    obs = env.reset()
    for _ in range(100):
        action = env.action_space.sample()
        print(action.shape)
        print(obs.shape)
        obs, rew, done, info = env.step(action)