# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**LUNA** (Learning with Uncertainty to Perform Neural Active Mapping) is a hierarchical RL system for active 3D scene reconstruction. An agent navigates a Gibson/MP3D environment, guided by a global policy that uses neural implicit map uncertainty to decide where to explore next.

Conda environment: `lena` (Python 3.9 required for Habitat compatibility).

## Key Commands

### Training and Evaluation

```bash
# Train the LUNA (lena) model
python main.py --config ./configs/train_lena.txt

# Train baseline (active neural SLAM)
python main.py --config ./configs/train_NSLAM.txt

# Evaluate on a specific scene
python main.py --config ./configs/eval_lena_Annawan.txt

# Run multiple experiments in parallel (edit run_experiments.py first)
python run_experiments.py

# Get list of all validation episode IDs
python utils/get_episodes.py
```

### Evaluation and Visualization

```bash
# Evaluate reconstruction quality (accuracy, completion, completion ratio)
python eval/auto_eval.py --config eval/eval_basic.yaml

# Plot results vs. uncertainty and scene coverage
cd eval && python plot_eval_results.py --eval evaluation_results --exp <experiment_name>
```

## Three Configuration Files

Every run is controlled by **three separate configs**:

| File | Purpose |
|------|---------|
| `configs/train_lena.txt` | Main args for `main.py` (GPU, episodes, policy arch, reward, paths) |
| `env/habitat/configs/gibson.yaml` | Habitat simulation (scene paths, sensor specs, agent params) |
| `env/habitat/configs/mapping.yaml` | Neural implicit map (hash grid, ensemble size, NeRF iters, mesh saving) |

Pass a custom mapping config with `--mapping_config <path>`. `run_experiments.py` auto-generates temporary copies to allow parallel experiments.

**Important config flags in `train_lena.txt`:**
- `global_arch = lena` — uses `Global_Policy_Tensor` with `TensorViewTransformer`; `nslam` uses plain CNN
- `use_NeRF_mapping = 1` — enables CoSLAM neural implicit mapping
- `use_uncertainty_reward = 1` — reward agent for reducing map uncertainty
- `heuristic_strategy = none|base|probe` — unsticking behavior

## Architecture

The system is hierarchical: a **global policy** sets long-term goals (every 25 local steps), while a **local policy** executes low-level actions toward them. A **mapping module** maintains the 3D neural implicit map.

```
main.py (master loop)
├── MappingHandler (utils/mapping_handler.py)
│   ├── Neural_SLAM_Module (model.py) — ResNet-18 encoder, depth projection, pose estimation
│   ├── CoSLAM neural implicit map — hash grid + ensemble uncertainty (env/habitat/exploration_env.py)
│   └── FMM planner — converts goal predictions to waypoints
│
├── GlobalPolicyHandler (utils/global_policy_handler.py)
│   ├── Global_Policy / Global_Policy_Tensor (model.py) — CNN or TensorViewTransformer on 8-ch map
│   └── PPO (algo/ppo.py) — trains global policy every 40 global steps
│
└── LocalPolicyHandler (utils/local_policy_handler.py)
    └── DdppoPolicy (model.py) — pretrained DD-PPO local navigation policy
```

**Per-step flow:**
1. Local policy executes action → Habitat sim returns RGB-D obs
2. Mapping module updates occupancy map and neural implicit map
3. Every 25 local steps: global policy gets new map input → predicts next long-term goal
4. Every 40 global steps: PPO updates global policy weights

**Episode lifecycle and reset semantics (important):**
- `envs.reset()` is called once per outer `ep_num` iteration in `main.py:61`. This is the **only** reset point.
- Per-env auto-reset is **disabled**: `exploration_env.step()` sets `self.habitat_env._episode_over = False` (`exploration_env.py:407`) so Habitat does not reset on `done=True`.
- When `done=True` fires (from `time >= max_episode_length` or `accumulated_ratio >= finish_ratio`), the env keeps stepping in its terminated state for the rest of the inner loop. Because `accumulated_ratio` only grows, a `finish_ratio`-triggered termination causes `finish_bonus` to fire every subsequent global step.
- Therefore `g_handler.g_masks == 0` at `process_rewards` time can be *first* termination or *N-th* re-termination — they are indistinguishable from `g_masks` alone. Aggregations over envs (logging, accumulators) must track first termination separately (see `GlobalPolicyHandler.episode_done`, cleared in `reset_episode`).

**Map input to global policy** is an 8-channel tensor:
`[obstacle_map, explored_map, current_pos, past_goal, map_boundary, uncertainty_map (×3)]`

## Key Files

- `main.py` — master training/eval loop; entry point for everything
- `model.py` — all neural network definitions (`Global_Policy`, `Global_Policy_Tensor`, `Neural_SLAM_Module`, `DdppoPolicy`, `RL_Policy`)
- `arguments.py` — all CLI arguments and defaults
- `env/habitat/exploration_env.py` — single-process env: `reset()`, `step()`, CoSLAM integration
- `env/__init__.py` — `make_vec_envs()` and `VecPyTorch` for parallel multi-scene execution
- `utils/global_policy_handler.py` — `GlobalPolicyHandler`: act, train PPO, save/load model
- `utils/mapping_handler.py` — `MappingHandler`: map updates, local/global map extraction
- `utils/local_policy_handler.py` — `LocalPolicyHandler`: local DD-PPO planning
- `mvt/tensor_view_transformer.py` — `TensorViewTransformer` used by `Global_Policy_Tensor`
- `algo/ppo.py` — PPO update step

## Ensemble Uncertainty

Epistemic uncertainty is estimated via **deep ensembles** (`ensemble_size=3` in `mapping.yaml`). Three independently initialized NeRF networks predict the same scene; empirical variance across predictions is used as the uncertainty signal. This is fed into the global policy map input and optionally used as a reward. See `gemini_instructions/uncertainty/GEMINI.md` for theory and implementation details.

## Outputs

- Models saved to: `<dump_location>/models/<exp_name>/`
- Logs/tensorboard: `<dump_location>/models/<exp_name>/`
- Images/videos: `<dump_location>/<exp_name>/`
- Meshes: controlled by `[mesh][vis]` in `mapping.yaml` (e.g., `vis: 10` saves every 10 iters)
- Results JSON: `eval/evaluation_results.json`

## Dataset Setup

After downloading, update `data_path` and `scenes_dir` in `env/habitat/configs/gibson.yaml` to point to your local Gibson dataset and pointnav_gibson_v1 task dataset paths.

Pretrained DD-PPO local policy: `pretrained_models/gibson-4plus-mp3d-train-val-test-resnet50.pth`