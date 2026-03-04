*  Currently, `reset` in @env/habitat/exploration_env.py is called automatcally when `done == True`. Change it so that it only resets when it is called manually.
* In @main.py, call `reset` when a new episode starts. 
    * do `m_handler._init_map_and_pose()`, `m_handler.get_full_map_pose`, `m_handler.get_local_map_pose`, `m_handler.get_global_input`, `g_handler.init_rollout`, and `g_handler.act(0)` at the start of an episode (between two for loops).
    * 