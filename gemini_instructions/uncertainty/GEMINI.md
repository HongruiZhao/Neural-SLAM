# Current Implementation of Ensemble Uncertainty
**Last Updated**: Monday, January 26, 2026

## Theory for uncertainty 
* We follow the paper "Simple and Scalable Predictive Uncertainty Estimation using Deep Ensembles" (@./gemini_instructions/uncertainty/deep_ensemble.pdf) for our ensemble uncertainty implementation.
* **Empirical Variance Implementation**:
    * According to the paper (Section 3.2), a common heuristic is to use an ensemble of NNs trained to minimize MSE.
    * The uncertainty is estimated using the **empirical variance** of the ensemble predictions (epistemic uncertainty).
    * Given $M$ ensemble members where each model $m$ predicts a value $\mu_{\theta_m}(x)$ (e.g., SDF or RGB value), the ensemble mean is $\mu_*(x) = \frac{1}{M} \sum_{m=1}^M \mu_{\theta_m}(x)$.
    * The empirical variance is calculated as:
      $$ \sigma^2_{emp}(x) = \frac{1}{M} \sum_{m=1}^M (\mu_{\theta_m}(x) - \mu_*(x))^2 $$
    * This captures the disagreement among the ensemble members, which serves as a proxy for the model's uncertainty about the prediction.
* **Training Methodology**:
    * **Objective**: For the empirical variance approach, each network in the ensemble is trained independently to minimize the Mean Squared Error (MSE) (or similar reconstruction loss) on the training data.
    * **Diversity**: Diversity among ensemble members is achieved through random initialization of the neural network parameters and the stochasticity of the training process (e.g., random data shuffling).
    * *Note*: The paper also proposes a more advanced method using Negative Log Likelihood (NLL) and adversarial training to learn calibrated predictive uncertainty (including heteroscedastic noise), but our current implementation focuses on the simpler empirical variance from MSE-trained networks.
* Currently, we rely solely on this empirical variance and do not explicitly model the heteroscedastic noise (learned variance $\sigma^2_{\theta_m}$) for each member.

## Implementation Overview
* The ensemble and the uncertainty grid are defined in the class `HashUncertainty` in `env/habitat/model/encodings.py`.
* The uncertainty grid uses an **Exponential Moving Average (EMA)** to fuse ensemble disagreement over time.
    * **Update Rule**: $U_{t} = \alpha \cdot U_{obs} + (1 - \alpha) \cdot U_{t-1}$ (where $\alpha = 0.1$).
    * **Initialization**: 
        * If `[grid][initial_uncert]` is set to a float value (e.g., `1e-2`), the grid is initialized to this constant.
        * If `[grid][initial_uncert]` is set to `null` (None), the grid is initialized dynamically. It starts at 0, and upon the first update, the **maximum variance** observed in that update batch is used to fill the entire grid. This effectively sets the "unobserved" background uncertainty to the highest uncertainty seen in the first frame.
    * **Manual Re-initialization**:
        * To control the initial variance of the ensemble members (which promotes diversity), we can manually re-initialize the `tcnn.Encoding` parameters.
        * This is controlled by the `[grid][custom_init]` (bool) and `[grid][init_gain]` (float) flags in the config.
        * If `custom_init` is True, each member's parameters are re-initialized using `torch.nn.init.xavier_normal_` with the specified `init_gain`. 
   
* The ensemble logic is handled in `JointEncoding` within `env/habitat/model/scene_rep.py`.
* A flag `self.if_extract_mesh` in `JointEncoding` controls the behavior between training (independent members) and inference (averaged output).
* A flag `multi_decoder` in `mapping.yaml` (under `grid`) toggles between using a shared decoder (False) or independent decoders for each ensemble member (True). When True, each member has its own `ColorSDFNet_v2`.


## Training Behavior (`if_extract_mesh = False`)
* **Goal**: Train each member of the ensemble independently to preserve diversity.
* **Forward Pass**:
    * `query_color_sdf` in @env/habitat/model/scene_rep.py returns concatenated outputs of shape `[N_rays * N_ensemble, N_samples, 4]`.
    * `target_rgb` and `target_d` in `forward()` are repeated `N_ensemble` times to match the concatenated output.
    * `z_vals` in `render_rays` are also repeated to match the expanded batch size.
* **Uncertainty Update**: The uncertainty grid is updated using the variance of the ensemble predictions with an **Exponential Moving Average (EMA)** rule.
    * **Conditional Update**: To ensure the ensemble provides stable uncertainty estimates, the grid is **only updated at the last iteration** of the mapping optimization loops (specifically in `first_frame_mapping` and `global_BA`). This prevents the uncertainty map from being polluted by unconverged predictions during the intermediate steps of optimization. We can allow the uncertainty always be able to updated by setting the `[grid][uncert_always_update]` flag in @env/habitat/configs/mapping.yaml.

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