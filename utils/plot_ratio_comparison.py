import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

runs = {
    'NeuralSLAM (Ours)': PROJECT_ROOT / 'results/dump/May7_evalNSLAMOurs_Oyens/accumulated_ratios.json',
    'NeuralSLAM (baseline)': PROJECT_ROOT / 'results/dump/May6_evalNSLAM_Oyens/accumulated_ratios.json',
}

plt.figure(figsize=(8, 5))
for label, path in runs.items():
    with open(path, 'r') as f:
        data = json.load(f)
    # Episode "0" -> shape: [num_global_steps, num_processes]
    ratios = np.asarray(data['0'])
    mean_over_envs = ratios.mean(axis=1)
    std_over_envs = ratios.std(axis=1)
    steps = np.arange(len(mean_over_envs))
    line, = plt.plot(steps, mean_over_envs, label=label, linewidth=2)
    plt.fill_between(steps,
                     mean_over_envs - std_over_envs,
                     mean_over_envs + std_over_envs,
                     alpha=0.15, color=line.get_color())

plt.xlabel('Global step')
plt.ylabel('Accumulated area-coverage ratio (mean over envs)')
plt.title('Coverage ratio vs. step — Oyens eval')
plt.legend(loc='lower right')
plt.grid(True, alpha=0.3)
plt.tight_layout()

out = PROJECT_ROOT / 'results/dump/ratio_comparison_May6_Oyens.png'
plt.savefig(out, dpi=150)
print(f'Saved {out}')