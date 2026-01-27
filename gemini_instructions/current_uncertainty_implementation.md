# Current Implementation of Ensemble Uncertainty
**Last Updated**: Monday, January 26, 2026

## Overview
* The ensemble and the uncertainty grid are defined in the class `HashUncertainty` in @env/habitat/model/encodings.py.
* The ensemble logic is handled in `JointEncoding` within @env/habitat/model/scene_rep.py.
* A flag `self.if_extract_mesh` in `JointEncoding` controls the behavior between training (independent members) and inference (averaged output).

## Training Behavior (`if_extract_mesh = False`)
* **Goal**: Train each member of the ensemble independently to preserve diversity.
* **Forward Pass**:
    * `query_color_sdf` in @env/habitat/model/scene_rep.py returns concatenated outputs of shape `[N_rays * N_ensemble, N_samples, 4]`.
    * `target_rgb` and `target_d` in `forward()` are repeated `N_ensemble` times to match the concatenated output.
    * `z_vals` in `render_rays` are also repeated to match the expanded batch size.
* **Uncertainty Update**: The uncertainty grid IS updated during training using the variance of the ensemble predictions computed internally in `query_color_sdf`.
* **Smoothness Loss**:
    * `query_sdf` returns expanded embeddings/SDFs.
    * `smoothness` in @env/habitat/ramen_mapping.py calculates Total Variation (TV) loss for each ensemble member separately and averages the results.

## Inference Behavior (`if_extract_mesh = True`)
* **Goal**: Extract a clean mesh and consistent color using the consensus of the ensemble.
* **Triggered by**: `save_mesh` in @env/habitat/ramen_mapping.py sets `self.model.if_extract_mesh = True`.
* **Forward Pass**:
    * `query_color_sdf` and `query_sdf` return the **mean** (average) of the ensemble outputs (shape `[N_rays, N_samples, ...]`).
    * The uncertainty grid is **NOT** updated during this phase.

## Key Files & Functions
* **@env/habitat/model/scene_rep.py**:
    * `JointEncoding.__init__`: Initializes `self.if_extract_mesh = False`.
    * `query_color_sdf`: Switches between concatenation (training) and averaging (inference).
    * `render_rays`: Handles `z_vals` repetition.
    * `forward`: Handles target repetition.
* **@env/habitat/ramen_mapping.py**:
    * `save_mesh`: Toggles `if_extract_mesh` to `True` before extraction and back to `False` after.
    * `smoothness`: Adapts to 5D input `[x, y, z, E, d]` for independent regularization.