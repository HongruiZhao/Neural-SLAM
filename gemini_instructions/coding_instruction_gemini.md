# Instructions for code implementation

## Improve evaluation 

* I am going to make two changes to @eval/eval_recon.py.
    * change evaluation metrics from `accuracy`, `completion`, and `completion_ratio` to `Precision 5cm`, `Recall 5cm`, and `F1 5cm`.
    * For every mesh evaluated by @eval/eval_recon.py, save visualizations of the three metrics. 


## New evaluation metrics
* `Precision 5cm`:  the percentage of reconstructed surface points whose nearest ground-truth point lies within a 5 cm threshold.
* `Recall 5cm`: the percentage of ground-truth points whose nearest reconstructed point lies within a 5 cm threshold.
* `F1 5cm`: the harmonic mean of `Precision 5cm` and `Recall 5cm`.


## Visualization of evaluation metrics
* For each visualization, we create a 2D grid whose resolution matches the `uncert_map` in @env/habitat/exploration_env.py.
* `Precision 5cm` plot: we find all the reconstructed surface points fall within each cell of the grid and compute their `Precision 5cm`.
* `Recall 5cm` plot: we find all the ground-truth points fall within each cell of the grid and compute their `Recall 5cm`.
* `F1 5cm` plot: elemetwise harmonic mean of the `Precision 5cm` plot and `Recall 5cm` plot.
* Combine three plots into one subplot with three columns and one row. Make sure their origins are properly set to match the uncertainty map visualization in @env/habitat/exploration_env.py.
* Create a foler named after the exp name (second last directory name of `rec_mesh` or `ckpt_path` in @eval/eval_recon.py). And save the subplot into it named after the mesh name. 
* Create a new Python script under @eval allow me to generate a video of all subplots saved under the same folder.


## Change `Replay` behavior 
* Currently, setting `Replay: False` in @env/habitat/configs/mapping.yaml completetly disable replay from keyframe database in @env/habitat/ramen_mapping.py.
* I want to add a new flag `uncert_replay` in `grid` of @env/habitat/configs/mapping.yaml, and its behaviors should be:
    * `uncert_replay: False`: `sdf_variance` are only computed for the rays sampled from `current_rays`, not those sampled from `keyframeDatabase`.
    * `uncert_replay: True`: `sdf_variance` from all available rays (current behavior).