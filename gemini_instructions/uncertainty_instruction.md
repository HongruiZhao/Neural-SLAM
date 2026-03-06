# Predictive uncertainty
* I want you to follow @gemini_instructions/uncertainty/deep_ensemble.pdf to implement predictive variance for the ensemble uncertainty.
* We enable predictive variance when `['grid']['uncertainty'] = 'ensemble_predictive'` in @env/habitat/configs/mapping.yaml.
* We only do predictive variance for the sdf value predictions.
* The predictive variance implementation has three main components:
    * **Output**: Each member of the ensemble outputs a mean and a variance for the prediction. This means we need to add to the output of the SDFNet in @env/habitat/model/decoder.py. Remeber to pass the output through the softplus function to enfore stricit positivity.
    * **Uncertainty Update**: 
        * For prediticve variance, we will have aleatoric uncertainty $\sum \sigma^2_m$ and the epistemic uncertainty $\sum \mu^2_m - \bar{\mu}^2$, where $\sigma^2_m$ and $\mu^2_m$ are the variance and the mean of each member, and $\bar{\mu}^2$ is the mean of the ensemble. 
        * Our uncertainty grid will become 4D $(\text{N}_x, \text{N}_y, \text{N}_z, 2)$, so each element in the grid stores the aleatoric uncertainty and the epistemic uncertaint for the position.
        * give a flag to allow visualizing aleatoric or epistemic uncertainty.
    * **Learning**: the variance output by each model needs to be added to the loss function for learning.
 
