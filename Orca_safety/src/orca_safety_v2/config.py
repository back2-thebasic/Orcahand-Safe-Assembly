from dataclasses import dataclass
from pathlib import Path
import numpy as np
import yaml


@dataclass(frozen=True)
class FilterConfig:
    safe_distance: float = 0.005
    activation_distance: float = 0.015
    cbf_eta: float = 0.2
    finite_difference_epsilon: float = 0.0001
    regularization: float = 1e-6
    max_step_rad: float | None = None
    solver_tolerance: float = 1e-7
    max_iterations: int = 4000
    nonlinear_backtracking_steps: int = 8

    def __post_init__(self):
        vals = (self.safe_distance, self.activation_distance, self.cbf_eta,
                self.finite_difference_epsilon, self.regularization, self.solver_tolerance)
        if not np.all(np.isfinite(vals)):
            raise ValueError('Parameters must be finite')
        if not 0 <= self.safe_distance < self.activation_distance:
            raise ValueError('Require activation_distance > safe_distance >= 0')
        if not 0 < self.cbf_eta <= 1 or self.finite_difference_epsilon <= 0:
            raise ValueError('Invalid eta or finite difference epsilon')
        if self.regularization < 0 or self.solver_tolerance <= 0 or self.max_iterations < 1:
            raise ValueError('Invalid QP parameters')
        if self.max_step_rad is not None and (not np.isfinite(self.max_step_rad) or self.max_step_rad <= 0):
            raise ValueError('max_step_rad must be positive')
        if not isinstance(self.nonlinear_backtracking_steps, int) or not 0 <= self.nonlinear_backtracking_steps <= 16:
            raise ValueError('nonlinear_backtracking_steps must be an integer from 0 to 16')


def load_config(path=None):
    path = Path(path) if path else Path(__file__).parent/'configs/v1_right.yaml'
    obj = yaml.safe_load(path.read_text())['collision_safety']
    return FilterConfig(**obj['parameters']), obj['pairs']
