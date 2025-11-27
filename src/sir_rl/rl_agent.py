import numpy as np
from typing import Dict, Tuple
from .sir_sim import build_population, SUS, INF
from .interventions import apply_intervention_step, cost_step

actions = np.array([0.0, 0.25, 0.5, 0.75, 1.0])

def discretize_state(S_frac: float, I_frac: float, t_frac: float, n_bins: int=8, t_bins:int=8) -> Tuple[int,int,int]:
    s_idx = int(np.clip(np.floor(S_frac * n_bins), 0, n_bins-1))
    i_idx = int(np.clip(np.floor(I_frac * n_bins), 0, n_bins-1))
    t_idx = int(np.clip(np.floor(t_frac * t_bins), 0, t_bins-1))
    return s_idx, i_idx, t_idx

def train_q_learning(N=1000, I0=5, beta=0.15, gamma=1/7, C=8, dt=1.0, T=150,
                     n_episodes=300, alpha=0.1, gamma_q=0.99, eps_start=0.2, eps_end=0.01,
                     n_bins=6, t_bins=8, lambda_epi=1.0, lambda_soc=0.1, intervention="contacts", seed=42):
    Q = np.zeros((n_bins, n_bins, t_bins, len(actions)))
    eps = eps_start
    history = []
    for ep in range(n_episodes):
        np.random.seed(seed + ep)
        states = build_population(N, I0)
        total_reward = 0.0
        for t_step in range(T):
            S_frac = (states == SUS).sum() / N
            I_frac = (states == INF).sum() / N
            t_frac = t_step / T
            s_idx, i_idx, time_idx = discretize_state(S_frac, I_frac, t_frac, n_bins, t_bins)
            if np.random.rand() < eps:
                a_idx = np.random.randint(len(actions))
            else:
                a_idx = int(np.argmax(Q[s_idx, i_idx, time_idx]))
            u = float(actions[a_idx])
            next_states, new_inf = apply_intervention_step(states, beta, gamma, C, dt, u, intervention)
            reward = -cost_step(new_inf, u, lambda_epi=lambda_epi, lambda_soc=lambda_soc)
            total_reward += reward
            S_frac_n = (next_states == SUS).sum() / N
            I_frac_n = (next_states == INF).sum() / N
            t_frac_n = min((t_step+1)/T, 0.999)
            s_n, i_n, time_n = discretize_state(S_frac_n, I_frac_n, t_frac_n, n_bins, t_bins)
            best_next = Q[s_n, i_n, time_n].max()
            Q[s_idx, i_idx, time_idx, a_idx] += alpha * (reward + gamma_q * best_next - Q[s_idx, i_idx, time_idx, a_idx])
            states = next_states
            if (states == INF).sum() == 0:
                break
        eps = max(eps_end, eps - (eps_start - eps_end) / n_episodes)
        history.append(total_reward)
    policy_idx = np.argmax(Q, axis=-1)
    return {"Q": Q, "policy_idx": policy_idx, "history": np.array(history)}