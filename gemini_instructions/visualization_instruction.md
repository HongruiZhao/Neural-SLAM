# Improve visualization during the evaluation 
* Now when `uncert_map` is not None, we display the "Uncertainty Mean vs Step" on the second row of the plot in @env/habitat/utils/visualizations.py.
* I want to add two more subplots to the second row:
    * **reward per stgep**: this is returned by `self.get_global_reward()` in @env/habitat/exploration_env.py and the latest value is saved into `self.info['exp_reward']`. You may want to create a new list at `reset` to store the history of per step reward. Keep in mind that the reward is obatined every `args.num_local_steps`, in comparison `self.uncert_sum_history` is saved for every step. So you may want to also save the timestep along with the reward to plot them correctly. 
    * **value function output**: this is returned as `g_value` in @main.py. You can save them into a list and pass it to `visualize_map` defined in in @utils/utils_for_main.py. Keep in mind that the `g_value` is also obatined every `args.num_local_steps`.
