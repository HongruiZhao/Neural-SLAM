* Add a new reward to `get_global_reward` in @env/habitat/exploration_env.py such that:
    * It penalizes the L1 distance of the agent current location to the `global_goals` output by the global policy. 
    * In other words, it encourages the global policy to output a global waypoint that is reachable within the limited local steps.
    * The `global_goals` is defined in local window. To transform it into gloabl coordinate in meters, refer to `plan` in @model.py.
    * Normalize the reward (maybe w.r.t to the size of local window), and give it a coefficient. 