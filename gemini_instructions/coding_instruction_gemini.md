# Instructions for code implementation

## Parallelize Ensemble 
* Use pytorch `vmap` function to parallelize ensemble in @env/habitat/model/encodings.py when we are not using heterogeneous ensemble.