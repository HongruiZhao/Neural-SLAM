# Instructions for code implementation

## Heterogeneous ensemble:
* Add a flag `heterogeneous` under `grid` in @env/habitat/configs/mapping.yaml.
* When `heterogeneous: True`, we want a heterogeneous ensemble:
    * In @env/habitat/model/encodings.py for each member of the ensemble, we will have different `n_levels`, `level_dim`, `base_resolution`, and `desired_resolution` for each ensemble.
    * `n_levels * n_dims` needs to be the same for all memebers of the ensemble so that they can have the same output dimension.
    *  `log2_hashmap_size` is fixed since it limits the max hash table size.
