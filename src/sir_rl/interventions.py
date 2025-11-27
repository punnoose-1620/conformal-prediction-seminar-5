from typing import Tuple
import numpy as np
from .sir_sim import step_sir, SUS, INF, REC

def apply_intervention_step(states: np.ndarray, beta: float, gamma: float, C: int, dt: float, u: float,
                            intervention: str = "contacts") -> Tuple[np.ndarray,int]:
    """
    Wrap one SIR step with intervention u in [0,1].
    - 'contacts': reduce contacts C_eff = round((1-u)*C)
    - 'transmission': reduce beta -> (1-u)*beta
    """
    u = float(np.clip(u, 0.0, 1.0))
    if intervention == "contacts":
        C_eff = max(0, int(round((1.0 - u) * C)))
        return step_sir(states, beta, gamma, C_eff, dt)
    elif intervention == "transmission":
        beta_eff = (1.0 - u) * beta
        return step_sir(states, beta_eff, gamma, C, dt)
    else:
        raise ValueError(f"Unknown intervention type: {intervention}")

def cost_step(new_infections: int, u: float, lambda_epi: float = 1.0, lambda_soc: float = 0.1) -> float:
    """Per-step cost: epidemiological penalty + social cost (quadratic in u)."""
    return lambda_epi * new_infections + lambda_soc * (u ** 2)