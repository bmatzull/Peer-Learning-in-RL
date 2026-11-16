import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

NPZ_DIR = Path("npz")
OUT_DIR = Path("plots")
OUT_DIR.mkdir(exist_ok=True)

SEEDS = (1, 2, 3)

CONDITIONS = {
    "ratio_intermittent_1v3": ("1 Intermittent vs 3 DQN", "#FF9800"),
    "ratio_intermittent_3v1": ("3 Intermittent vs 1 DQN", "#FF6600"),
    "ratio_reformed_1v3":     ("1 Reformed vs 3 DQN",     "#4CAF50"),
    "ratio_reformed_3v1":     ("3 Reformed vs 1 DQN",     "#2E7D32"),
    "ratio_critical_state_1v3": ("1 Critical State vs 3 DQN", "#F44336"),
    "ratio_critical_state_3v1": ("3 Critical State vs 1 DQN", "#B71C1C"),
}

def load_conditions(env_prefix, condition_key, subdir="Ratio"):
    per_seed = []
    for seed in SEEDS:
        fname = NPZ_DIR /  subdir / f"{env_prefix}{condition_key}_seed{seed}_evaluations.npz"
        d = np.load(fname)
        per_seed.append((d["timesteps"], d["results"].mean(axis=1)))
    min_len = min([r.shape[0] for _, r in per_seed])
    timesteps = per_seed[0][0][:min_len]
    results = np.array([r[:min_len] for _, r in per_seed])
    return timesteps, results

def plot_env(env_prefix, title, subdir="Ratio"):
    fig, ax = plt.subplots(figsize=(10, 5))
    for key, (label, color) in CONDITIONS.items():
        timesteps, results = load_conditions(env_prefix, key, subdir="Ratio")
        mean = results.mean(axis=0)
        std = results.std(axis=0)
        ax.plot(timesteps / 1e6, mean, label=label, color=color)
        ax.fill_between(timesteps / 1e6, mean - std, mean + std, alpha=0.15, color=color)
        ax.set_xlim(0, 0.5)
    ax.set_title(title)
    ax.set_xlabel("Timesteps (M)")
    ax.set_ylabel("Mean Reward")
    ax.legend()
    plt.savefig(OUT_DIR / f"{title}.png")
    plt.close()

plot_env("", "Room-v21-ratio", subdir="Ratio")
plot_env("fl_", "FrozenLake-ratio", subdir="Ratio")