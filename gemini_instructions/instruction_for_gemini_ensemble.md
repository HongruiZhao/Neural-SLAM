# The current implementation of ensemble 
* The ensemble and the uncertainty grid are defined in the class `HashUncertainty` in @env/habitat/model/encodings.py.
* The ensemble outputs are called in function `query_sdf` and `query_color_sdf` in @env/habitat/model/scene_rep.py. 
* Currently, the outputs of the ensemble are averaged out. The averaged output is then returned by `query_sdf` and `query_color_sdf`.
* Loss computations for the outputs of `query_color_sdf` happen in `forward` of @env/habitat/model/scene_rep.py
* Loss computations for the outputs of `query_sdf` happen in `smootheness` in @env/habitat/ramen_mapping.py
* Configuration file is @env/habitat/configs/mapping.yaml.

# Training and inference 
* **training**: When the neural implicit map is being updated in `first_frame_mapping` and `global_BA` of @env/habitat/ramen_mapping.py
* **inference**: When meshed are being extracted by `save_mesh` in @env/habitat/ramen_mapping.py.

# General rules
* Use einstein operations like `torch.einsum` and `einops.rearrange` as much as possible to improve readbility. 
* Update Args-Reutrns block comments for functions when they are modified. 
* Restrain from writing too many comments in the codes. 
* If we change the name of a config flag in the codes, it also needs to be changed in the configuration file.
* I will provide you with an initial plan for implementing the changes, 

# Plan for implementing the changes 
* Training an ensemble with their avergaed reuslts leads to dependencies between the memebers of the ensemble. For example, we can have one memeber output the correct predictions and the other two memebers output wrong but opposite ouputs that cancel each other out. As a result, we still have the correct averaged output, but the variance of the ensemble is not a good estimation of uncertainty anymore. 
* Thus we want to train each memeber of the ensemble independently, and only average their predictions during inference. 
## Training
* During training, instead of returning the average output (shape is [N_rays, N_samples]), concatenate all the outputs of the ensemble (shape is [N_rays*N_ensemble, N_samples]).
* `target_d` and `target_rgb` in `forward()` of @env/habitat/model/scene_rep.py need to be repeated accoradingly to match the concatenate. use `einops.repeat` to do so.
* `z_vals` in `render_rays` in @env/habitat/model/scene_rep.py also needs to be repeated to match the concatenated outputs before `raw2outputs`.
* `smoothness` in @env/habitat/ramen_mapping.py needs to be changed to accomendate concatenated outputs. It should calculate TV loss for each ensemble member separately and then sum/average them.
## Inference    
* We already have a flag `self.model.do_update_uncert = False` in `save_mesh` of @env/habitat/ramen_mapping.py to inform the model we are at the inference, Change the name of the flag from `do_update_uncert` to `if_extract_mesh`.
* `if_extract_mesh=True`, we return the average result in `query_sdf` and `query_color_sdf` and we do not update the uncertainty grid. Otherwise we reutrn the concatenated results and do update the uncertainty grid.