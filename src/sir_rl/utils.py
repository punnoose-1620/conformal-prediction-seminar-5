import matplotlib.pyplot as plt
import numpy as np

def plot_sir(data: dict, title: str = "SIR dynamics"):
    t = np.arange(len(data["S"]))
    plt.figure()
    plt.plot(t, data["S"], label="Susceptible")
    plt.plot(t, data["I"], label="Infectious")
    plt.plot(t, data["R"], label="Recovered")
    plt.xlabel("Time (days)")
    plt.ylabel("Number of agents")
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.show()

def plot_training(history: np.ndarray, window: int = 20, title: str = "Training curve"):
    episodes = np.arange(len(history))
    if len(history) >= window:
        ma = np.convolve(history, np.ones(window)/window, mode='valid')
    else:
        ma = None
    plt.figure()
    plt.plot(episodes, history, alpha=0.4, label="Episode return")
    if ma is not None:
        plt.plot(np.arange(window-1, len(history)), ma, label=f"MA(window={window})", linewidth=2)
    plt.xlabel("Episode")
    plt.ylabel("Episode return")
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.show()