# RL Training with Uncertainty Summary

## Overview
The current RL training workflow is designed to train a global policy for active exploration using an ensemble-based uncertainty reward. The system uses PPO for the global policy and a local policy (or planner) for low-level navigation.

## Configuration
*   **Config File**: `configs/train_lena.txt`
*   **Key Flags**:
    *   `use_uncertainty_reward = 1`: Enables reward based on uncertainty reduction.
    *   `use_NeRF_mapping = 1`: Enables the Neural Implicit Mapping (NeRF-based) module.
    *   `train_global = 1`: Active training of the global policy.

## Workflow

### 1. Initialization (`main.py`)
*   The environment (`Exploration_Env`) is initialized.
*   `GlobalRolloutStorage` is set up to store observations, rewards, and actions.
*   The global policy (`RL_Policy`) outputs long-term goals.

### 2. Environment Step (`env/habitat/exploration_env.py`)
*   **Mapping**: At each step, the `nerf_mapper` updates the neural implicit map using RGB-D observations.
*   **Uncertainty Map**: The uncertainty map (`self.uncert_map`) is retrieved from the NeRF mapper (specifically `self.nerf_mapper.model.embed_fn.get_uncert_map()`).
*   **Reward Calculation**:
    *   The `get_global_reward()` function is called every `num_local_steps` (default 25).
    *   **Uncertainty Reward**: Calculated as the reduction in total uncertainty over the explorable area.
        *   `current_uncert_sum = (self.uncert_map * self.explorable_map).sum()`
        *   `m_reward = self.prev_uncert_sum - current_uncert_sum`
        *   The reward is scaled by `1e-4` to match the magnitude of area coverage rewards.
    *   **Area Coverage Reward** (Fallback): If uncertainty reward is disabled, it uses the increase in explored area.

### 3. Policy Update (`main.py`)
*   The global reward is collected from `infos['exp_reward']`.
*   Rewards are stored in `g_rollouts`.
*   PPO update is performed on the global policy using the collected rollouts.

## Key Observations
*   **Reward Scaling**: The uncertainty reward is scaled by a hardcoded factor of `1e-4`. This might need tuning to ensure effective learning.
*   **Uncertainty Source**: The uncertainty comes from the ensemble variance (implemented in `HashUncertainty` class).
*   **Performance**: The NeRF mapping runs every step, which is computationally intensive.

## Files Involved
*   `main.py`: Main training loop.
*   `env/habitat/exploration_env.py`: Environment logic, mapping integration, and reward calculation.
*   `env/habitat/ramen_mapping.py`: Interface for the NeRF mapping module.
