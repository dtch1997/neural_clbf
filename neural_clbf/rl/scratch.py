import gym
import d4rl
import neural_clbf.envs # noqa: F401

if __name__ == "__main__":
    env = gym.make("CAF-InvertedPendulum-v2")
    env.reset()
    
    # Test dataset
    dataset = d4rl.qlearning_dataset(env)

    # Sanity check env
    obs = env.reset()
    for _ in range(100):
        action = env.action_space.sample()
        print(action.shape)
        print(obs.shape)
        obs, rew, done, info = env.step(action)