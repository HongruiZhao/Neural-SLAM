import json 
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import configargparse

def main(args):

    with open(f'{args.eval}.json', 'r') as f:
        eval_results = json.load(f)
    f1 = eval_results['f1']
        
    with open(f'../results/dump/{args.exp}/accumulated_ratios.json', 'r') as f:
        all_accumulated_ratios = json.load(f)

    with open(f'../results/dump/{args.exp}/uncertainty_history.json', 'r') as f:
        all_uncertainty = json.load(f)

    chosen_episode_id = '0' 
    accumulated_ratio = np.array(all_accumulated_ratios[chosen_episode_id]).flatten()
    uncertainty = np.array(all_uncertainty[chosen_episode_id]).flatten()

    steps_comp_ratio = np.arange(0, 1000, 10)
    num_local_steps = 25
    max_episode_length = 1000
    steps_acc_ratio = np.arange(num_local_steps - 1, max_episode_length, num_local_steps)
    interp_acc_ratio = np.interp(steps_comp_ratio, steps_acc_ratio, accumulated_ratio)
    interp_uncertainty = np.interp(steps_comp_ratio, steps_acc_ratio, uncertainty)

    # Create a DataFrame for correlation
    df = pd.DataFrame({
        'F1 (5cm)': f1,
        'Scene Coverage Ratio': interp_acc_ratio,
        'Uncertainty Ratio': -interp_uncertainty # negative because uncertainty decreases instead of increasing
    })
    corr_matrix = df.corr(method='pearson')
    print("\n--- Correlation Matrix (Pearson) ---")
    print(corr_matrix)

    fig, ax1 = plt.subplots()
    # Plot completion ratio on the primary y-axis
    color = 'tab:red'
    ax1.set_xlabel('Step')
    ax1.set_ylabel('F1 (5cm)', color=color)
    ax1.plot(steps_comp_ratio, f1, color=color, label='F1 (5cm)')
    ax1.tick_params(axis='y', labelcolor=color)
    ax1.invert_yaxis()
    ax1.set_xticks(np.arange(0, 1000, 50))
    ax1.tick_params(axis='x', labelsize=6)

    # Create a second y-axis to plot accumulated ratio
    ax2 = ax1.twinx()
    color = 'tab:blue'
    ax2.set_ylabel('Scene Coverage Ratio', color=color)
    ax2.plot(steps_comp_ratio, interp_acc_ratio, color=color, label=f'Scene coverage Ratio (Ep {chosen_episode_id})')
    ax2.tick_params(axis='y', labelcolor=color)
    ax2.invert_yaxis()

    ax3 = ax1.twinx()
    ax3.spines["right"].set_position(("axes", 1.2))
    color = 'tab:green'
    ax3.set_ylabel('Uncertainty Ratio', color=color)
    ax3.plot(steps_comp_ratio, interp_uncertainty, color=color, label=f'Uncertainty (Ep {chosen_episode_id})')
    ax3.tick_params(axis='y', labelcolor=color)


    # Add legends
    lines, labels = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    lines3, labels3 = ax3.get_legend_handles_labels()
    ax2.legend(lines + lines2 + lines3, labels + labels2 + labels3, loc='upper right')

    plt.title(f'{args.exp} Evaluation Metrics')
    fig.tight_layout()
    plt.savefig(f'eval_{args.exp}.jpg', dpi=500)

if __name__ == "__main__":
    parser = configargparse.ArgumentParser()
    parser.add_argument('--eval', type=str, default='ensemble_Cantwell_Feb2')
    parser.add_argument('--exp', type=str)
    args = parser.parse_args()
    main(args)