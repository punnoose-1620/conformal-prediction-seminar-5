import numpy as np
from tqdm import tqdm
from .sir_sim import build_population
from .sir_sim import INF, SUS

def estimate_R0_empirical(beta: float, gamma: float, C: int, dt: float, N: int, repetitions: int = 100, seed: int = 0):
    sec_counts = []
    for rep in range(repetitions):
        np.random.seed(seed + rep)
        states = build_population(N, I0=1)
        index_idx = np.where(states == INF)[0][0]
        infected_by_index = set()
        while states[index_idx] == INF:
            possible = np.delete(np.arange(N), index_idx)
            contacts = np.random.choice(possible, size=min(C, N-1), replace=False)
            p_trans = 1 - np.exp(-beta * dt)
            for j in contacts:
                if states[j] == SUS and np.random.rand() < p_trans:
                    states[j] = INF
                    infected_by_index.add(j)
            # recoveries
            inf_idx = np.where(states == INF)[0]
            for i in inf_idx:
                if np.random.rand() < (1 - np.exp(-gamma*dt)):
                    states[i] = 2  # REC
        sec_counts.append(len(infected_by_index))
    return np.mean(sec_counts), np.std(sec_counts) / np.sqrt(repetitions)

def run_beta_sweep(betas, gamma, C, dt, N, repetitions=100):
    means, ses = [], []
    for b in tqdm(betas, desc="R0 sweep"):
        m, se = estimate_R0_empirical(b, gamma, C, dt, N, repetitions)
        means.append(m); ses.append(se)
    return np.array(means), np.array(ses)