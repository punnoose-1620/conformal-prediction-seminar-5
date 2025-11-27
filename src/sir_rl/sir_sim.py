import numpy as np
import random
from typing import Tuple, Dict

SUS, INF, REC = 0, 1, 2

def build_population(N: int, I0: int = 1) -> np.ndarray:
    """Return state array with I0 infectious, rest susceptible."""
    states = np.full(N, SUS, dtype=np.int8)
    infectious_idx = np.random.choice(N, size=I0, replace=False)
    states[infectious_idx] = INF
    return states

def step_sir(states: np.ndarray, beta: float, gamma: float, C: int, dt: float) -> Tuple[np.ndarray,int]:
    """
    One synchronous time step for well-mixed ABS:
    - every infectious agent makes up to C contacts (without replacement)
    - per-contact transmission prob p_trans = 1 - exp(-beta*dt)
    - per-step recovery prob p_rec = 1 - exp(-gamma*dt)
    Returns next_states and number of new infections this step.
    """
    N = len(states)
    p_trans = 1 - np.exp(-beta * dt)
    p_rec = 1 - np.exp(-gamma * dt)
    next_states = states.copy()
    new_infections = 0

    infectious_idx = np.where(states == INF)[0]
    for i in infectious_idx:
        possible = np.delete(np.arange(N), i)
        contacts = np.random.choice(possible, size=min(C, N-1), replace=False)
        for j in contacts:
            if states[j] == SUS and np.random.rand() < p_trans:
                next_states[j] = INF
                new_infections += 1
        if np.random.rand() < p_rec:
            next_states[i] = REC

    return next_states, new_infections

def run_sim(N: int, I0: int, beta: float, gamma: float, C: int, dt: float, T: int, seed: int = None) -> Dict:
    """Run T steps and return S/I/R time series and new_infections per step."""
    if seed is not None:
        np.random.seed(seed)
        random.seed(seed)
    states = build_population(N, I0)
    S_hist, I_hist, R_hist = [], [], []
    new_infections_per_step = []
    for t in range(T):
        S_hist.append(int((states == SUS).sum()))
        I_hist.append(int((states == INF).sum()))
        R_hist.append(int((states == REC).sum()))
        states, new_inf = step_sir(states, beta, gamma, C, dt)
        new_infections_per_step.append(new_inf)
        if (states == INF).sum() == 0:
            for _ in range(t+1, T):
                S_hist.append(int((states == SUS).sum()))
                I_hist.append(0)
                R_hist.append(int((states == REC).sum()))
                new_infections_per_step.append(0)
            break
    return {
        "S": np.array(S_hist), 
        "I": np.array(I_hist), 
        "R": np.array(R_hist), 
        "new_infections": np.array(new_infections_per_step)
        }