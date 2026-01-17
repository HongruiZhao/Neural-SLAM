import json 
import matplotlib.pyplot as plt
import numpy as np

def main():
    
    with open('evaluation_results_Cantwell.json', 'r') as f:
        eval_results = json.load(f)
    completion_ratio = eval_results['comp_ratio']
        
    with open('../results/dump/Cantwell_Jan15/accumulated_ratios.json', 'r') as f:
        all_accumulated_ratios = json.load(f)
    chosen_episode_id = '0' 
    accumulated_ratio = all_accumulated_ratios[chosen_episode_id]

    steps_comp_ratio = np.arange(0, 1000, 10)
    num_local_steps = 25
    max_episode_length = 1000
    steps_acc_ratio = np.arange(num_local_steps - 1, max_episode_length, num_local_steps)

    fig, ax1 = plt.subplots()

    # Plot completion ratio on the primary y-axis
    color = 'tab:red'
    ax1.set_xlabel('Step')
    ax1.set_ylabel('Completion Ratio', color=color)
    ax1.plot(steps_comp_ratio, completion_ratio, color=color, label='Completion Ratio')
    ax1.tick_params(axis='y', labelcolor=color)
    ax1.invert_yaxis()
    ax1.set_xticks(np.arange(0, 1000, 50))
    ax1.tick_params(axis='x', labelsize=6)

    # Create a second y-axis to plot accumulated ratio
    ax2 = ax1.twinx()
    color = 'tab:blue'
    ax2.set_ylabel('Accumulated Ratio', color=color)
    ax2.plot(steps_acc_ratio, accumulated_ratio, color=color, label=f'Accumulated Ratio (Ep {chosen_episode_id})')
    ax2.tick_params(axis='y', labelcolor=color)
    ax2.invert_yaxis()
    # Add legends
    lines, labels = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax2.legend(lines + lines2, labels + labels2, loc='upper right')

    plt.title('Evaluation Metrics')
    fig.tight_layout()
    plt.savefig('evaluation_combined.jpg', dpi=500)

if __name__ == "__main__":
    main()