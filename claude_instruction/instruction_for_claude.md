* In `feed_forward_generator` of @utils/storage.py, add codes to find env i and step t that masks[t,i] = 0 and masks[t-1,i] = 0
* These i and t are after an environment has been done, so we shouldn't use time for training PPO. 
* We only want to sample from the i and t that are not after an environment has been done, we can do:
    * Create a new boolean variable `after_done` with the same shape as `masks` to indicate what elements are after done.
    * Flatten all variables (`obs`, `rec_states` etc), and choose only the elements that are not after done.
    * These elements set the new batch_size, and the sampler only samples from these elements.

* Write a test code:
    * 10 steps, 3 environments. 
    * The 3 environments get done at iteration 5, 6, 7. 
    * To test if sampler works as intended. 