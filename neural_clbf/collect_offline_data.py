import torch
from neural_clbf.systems.control_affine_system import ControlAffineSystem
from neural_clbf.systems.inverted_pendulum import InvertedPendulum


def torch_random_uniform(lo: torch.Tensor, hi: torch.Tensor, size: torch.Size):
    """ Sample random values from a uniform distribution """
    return (hi - lo) * torch.rand(size) + lo

def sample_uniform_control(system: ControlAffineSystem, n_samples: int):
    """ Sample random controls """
    return torch_random_uniform(*system.control_limits, size=(n_samples, system.n_controls))

def step_system(system: ControlAffineSystem, x: torch.Tensor, u: torch.Tensor):
    """ Step the system forward """
    x_next = system.zero_order_hold(
        x, u, system.dt
    )
    return x_next


def collect_data(system: ControlAffineSystem, num_samples: int):
    states = system.sample_state_space(batch_size)
    controls = sample_uniform_control(system, batch_size)
    next_states = step_system(system, states, controls)
    is_safe = system.safe_mask(states)
    rewards = is_safe.float()
    dones = 1 - rewards
    return {
        "observations": states,
        "actions": controls,
        "next_observations": next_states,
        "rewards": torch.zeros(batch_size, 1),
        "terminals": dones,
    }


if __name__ == "__main__":

    batch_size = 1_000_000
    nominal_params = {"m": 1.0, "L": 1.0, "b": 0.01}
    system: ControlAffineSystem = InvertedPendulum(nominal_params=nominal_params)
    data = collect_data(system, batch_size)
    # Convert to numpy
    for k, v in data.items():
        data[k] = v.cpu().numpy()
    import pickle 
    with open("data.pkl", "wb") as f:
        pickle.dump(data, f)