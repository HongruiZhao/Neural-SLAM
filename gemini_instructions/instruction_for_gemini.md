# Implement vmap for the ensemble in @Neural-SLAM
## Hash grid 
* Currently, in @Neural-SLAM/env/habitat/model/encodings.py, we use tinycudann hash grid 
* We want to accelerate the ensemble computation using `vmap` (when the ensemble is not heterogeneous), which tinycudann does not support.
* Thus we need native pytorch implementation for the hash grid:
    *  When `tcnn_encoding:False` of the `grid` key in @Neural-SLAM/env/habitat/configs/mapping.yaml, we use native pytorch implementation instead of tinycudann and utilize `vmap` to run the ensemble in parallele in @Neural-SLAM/env/habitat/model/encodings.py.
    * The native pytorch hash grid is implementend in @Neural-SLAM/env/habitat/model/hash_encoding.py.
## Decoder 
* When `tcnn_network: False` of the `decoder` key in @Neural-SLAM/env/habitat/configs/mapping.yaml, we use pytorch MLP (this is already implemented in Neural-SLAM/env/habitat/model/decoder.py).
* Modify ensemble logic in @Neural-SLAM/env/habitat/model/scene_rep.py to use `vmap` for the decoder ensemble when `tcnn_network: False`.
## Verify 
Write a test code to verify your implementation.
## Documentation 
Summarize the new implementations in @Neural-SLAM/gemini_instructions/mapping_module/GEMINI.md