import torch 

from typing import Optional, List, Tuple
from simulation import (
    simulate,
    DynamicsCallable, 
    ControllerCallable, 
    MetricCallable, 
    MetricDerivCallable
)

def random_uniform(n: int, lower: torch.Tensor, upper: torch.Tensor, device):
    return torch.unsqueeze(upper - lower, 0) * torch.rand((n, ) + upper.shape, device=device)

def eval_contraction_metric(
    state_space: List[Tuple[float, float]],
    x_ref: torch.Tensor,
    u_ref: torch.Tensor,
    sim_dt: float,
    controller_dt: float,
    dynamics: DynamicsCallable,
    controller: ControllerCallable,
    metric: MetricCallable,
    metric_derivative: MetricDerivCallable,
    control_bounds: Optional[List[float]] = None,
    steps: int = 10,
):
    """ 
    Check whether a contraction metric holds within a given region
    for a fixed reference state, reference control, and tracking controller.
    
    x_ref: (n_state_dims,) tensor of reference state.
    u_ref: (n_control_dims,) tensor of reference control
    """
    
    metric_is_valid = True

    # Generate a grid of starting states within state_space
    axis_coords = []
    for si_lb, si_ub in state_space:
        coord = torch.linspace(si_lb, si_ub, steps=steps)
        axis_coords.append(coord)
    x_batch = torch.cartesian_prod(*axis_coords) # (steps ** d, d), d = n_state_dims

    # Evaluate whether metric is decreasing
    n_batch = x_batch.shape[0]
    x_ref_batch = x_ref.unsqueeze(0).repeat((n_batch, 1))
    u_ref_batch = u_ref.unsqueeze(0).repeat((n_batch, 1))
    dMdt = metric_derivative(x_batch, x_ref_batch, u_ref_batch) # (n_batch, 1)

    return x_batch, dMdt

