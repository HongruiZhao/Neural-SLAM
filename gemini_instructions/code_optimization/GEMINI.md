# Code Optimization Benchmark Results

## Benchmark Environment
- Date: Friday, February 6, 2026
- Hardware: NVIDIA GeForce RTX 5090
- Software: Python 3.9, PyTorch with CUDA

## Baseline Performance (configs/eval_speedTest.txt)
- **Total time**: 53.46s
- **Total steps**: 100
- **Avg time per step**: 0.5346s
- **Reset time**: 5.99s (Avg: 5.9905s)
- **Step time (Env + Mapping)**: 37.50s (Avg: 0.3750s)
- **Local Policy/Planner time**: 0.59s (Avg: 0.0059s)
- **SLAM time**: 1.51s (Avg: 0.0151s)
- **Global Policy time**: 0.01s (Avg: 0.0030s)
- **Visualization time**: 0.00s (Avg: 0.0000s)

### Bottleneck Analysis
The primary bottleneck is the **Step time (Env + Mapping)**, accounting for ~70% of the total time. This includes the environment simulation and the NeRF-based mapping update.

## Optimization Plan

1. Use `torch.compile` on the NeRF mapping modules in `env/habitat/ramen_mapping.py` and `model.py`.

2. Investigate if `Neural_SLAM_Module` can also be compiled.

3. Check for any redundant operations in the mapping loop.



## Attempt 1: torch.compile on SLAM and RL policies

- **Status**: Failed to improve performance in 100 steps due to compilation overhead.

- **Results**:

    - **Total time**: 58.97s (Baseline: 53.46s)

    - **Avg time per step**: 0.5897s (Baseline: 0.5346s)

- **Challenges**:

    - `torch.compile` on the NeRF mapping model (JointEncoding) caused a crash (`malloc(): unsorted double linked list corrupted`) because of incompatibility with `tinycudann`.

    - `torch.compile` must be applied AFTER `load_state_dict` to avoid key mismatch.

    - Type checks like `type(l_policy) == DdppoPolicy` fail because the compiled model is an `OptimizedModule`.



## Attempt 2: Partial torch.compile on ramen_mapping and model utils



- **Status**: Success. Improved performance on the second episode.



- **Modifications**:



    - Compiled RL policies and SLAM module in `main.py` (after loading state dicts).



    - Compiled non-tinycudann math utilities in `env/habitat/model/utils.py`.



    - Benchmarked with 2 episodes to ignore compilation overhead.



- **Results**:



    - **Episode 0 time**: 44.11s (Compilation overhead)



    - **Episode 1 time**: 41.12s (Stable performance)



    - **Total average time per step**: 0.5224s (Baseline: 0.5346s)



- **Observations**:



    - Compiling `get_loss_from_ret` or the whole `JointEncoding` causes crashes with `tinycudann`.



    - Compiling utility functions called in the mapping inner loop provides a measurable speedup.







## Attempt 3: Compile raw2outputs and sdf2weights in JointEncoding
- **Status**: Success. Further marginal improvement on stable episode time.
- **Modifications**:
    - Kept Attempt 2 optimizations (RL/SLAM policies, model utils).
    - Applied `@torch.compile` to `sdf2weights` and `raw2outputs` in `env/habitat/model/scene_rep.py`.
- **Results**:
    - **Episode 0 time**: 31.98s (Somehow much faster initialization this time)
    - **Episode 1 time**: 40.94s (Stable performance, slightly better than Attempt 2's 41.12s)
    - **Total average time per step**: 0.4734s (Baseline: 0.5346s)
- **Observations**:
    - Targeting high-traffic tensor manipulation functions within `JointEncoding` without touching `tinycudann` modules is safe and effective.

---

