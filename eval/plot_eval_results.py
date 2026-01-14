import json 
import matplotlib.pyplot as plt
import numpy as np

def main():
    
    with open('evaluation_results.json', 'r') as f:
        eval_results = json.load(f)

    completion_ratio = eval_results['comp_ratio']
    step = np.arange(0, 1000, 10)
    plt.xticks(np.arange(0, 1000, 50))
    plt.xticks(fontsize=6)
    plt.plot(step, completion_ratio)
    plt.xlabel('Step')
    plt.ylabel('Completion Ratio')
    plt.gca().invert_yaxis()
    plt.savefig('evaluation.jpg', dpi=500)

if __name__ == "__main__":
    main()