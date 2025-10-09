# Gemini Code Assistant Context

## Project Overview

This project is **Neural-SLAM**, an implementation of "learn to perform neural active mapping". It utilizes the [Habitat-sim](https://github.com/facebookresearch/habitat-sim) platform for training and evaluation of autonomous agents in 3D environments. The project is written in Python and uses PyTorch for its deep learning models.

The core of the project is a hierarchical reinforcement learning approach with three main components:

1.  **Neural SLAM Module:** This module is responsible for building a map of the environment and estimating the agent's pose. It uses a neural network to process visual observations and update its map and pose estimates.
2.  **Global Policy:** A high-level policy that takes the map from the Neural SLAM module as input and sets long-term goals for the agent.
3.  **Local Policy:** A low-level policy that takes the agent's current observation and the short-term goal from the global policy as input and outputs low-level actions (e.g., move forward, turn left, turn right).

The project is structured to allow for training and evaluation of these components in an end-to-end fashion.

## Building and Running

### Installation

The project uses a conda environment and requires several dependencies to be installed from source, including `habitat-sim`, `habitat-lab`, and `tiny-cuda-nn`. Detailed installation instructions are available in the `README.md` file.

### Training

To train the models, run the following command:

```bash
python main.py --config ./configs/train_NSLAM.txt
```

### Evaluation

To evaluate a trained model, run the following command:

```bash
python main.py --config ./configs/eval_vis.txt
```

The `eval_vis.txt` file can be modified to specify which saved model to evaluate.

## Development Conventions

*   **Configuration:** The project uses a combination of command-line arguments and configuration files.
    *   `arguments.py`: Defines the command-line arguments.
    *   `configs/`: Contains configuration files for training and evaluation.
    *   `env/habitat/configs/`: Contains configuration files for the Habitat simulation environment and the neural implicit map.
*   **Code Structure:**
    *   `main.py`: The main entry point for training and evaluation.
    *   `model.py`: Defines the neural network architectures for the global policy, local policy, and Neural SLAM module.
    *   `algo/`: Contains the implementation of the PPO reinforcement learning algorithm.
    *   `env/`: Contains the environment-related code, including the Habitat simulation environment.
    *   `utils/`: Contains utility functions for storage, optimization, and other tasks.
*   **Logging:** The project uses the `logging` module to log training progress to a file and `tensorboard` for visualization.
*   **Model Saving:** Models are saved periodically and when a new best performance is achieved. The saved models are stored in the `dump_location` specified in the arguments.
