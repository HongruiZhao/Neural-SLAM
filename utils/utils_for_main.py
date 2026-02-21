import numpy as np
import torch 
            

def run_local_planner(num_scenes, 
                     global_goals, global_input, planner_pose_inputs,
                     envs):
    """Run local planner 
    Args:
        num_scenes: 
        global_goals: num_scenes x global waypoints. in pixels (scaled by map_resolution), local window origin.
        global_input: num_scenes x 8 x local_w x local_h maps. 
        planner_pose_inputs: 
            - num_scenes x 7. 
            - 1-3 store continuous global agent x(m), y(m), o (deg). 4-7 store local map boundaries (gx1, gx2, gy1, gy2) 
        envs: 
    Returns:
        output:   
    """
    planner_inputs = [{} for e in range(num_scenes)]
    for e, p_input in enumerate(planner_inputs):
        p_input['goal'] = global_goals[e]
        p_input['map_pred'] = global_input[e, 0, :, :].detach().cpu().numpy()
        p_input['exp_pred'] = global_input[e, 1, :, :].detach().cpu().numpy()
        p_input['pose_pred'] = planner_pose_inputs[e]

    # Output stores local goals as well as the the ground-truth action
    output = envs.get_short_term_goal(planner_inputs)

    return output


def visualize_map(num_scenes, 
                     global_goals, global_input, planner_pose_inputs,
                     envs, full_map=None, local_map=None, heuristic=None, value_history=None):
    """for visualization
    Args:
        num_scenes: 
        global_goals: num_scenes x global waypoints. in pixels (scaled by map_resolution), local window origin.
        global_input: num_scenes x 8 x local_w x local_h maps. 
        planner_pose_inputs: 
            - num_scenes x 7. 
            - 1-3 store continuous global agent x(m), y(m), o (deg). 4-7 store local map boundaries (gx1, gx2, gy1, gy2) 
        envs: 
        full_map: num_scenes x 4 x full_w x full_h maps.
        local_map: num_scenes x 4 x local_w x local_h maps.
        heuristic: num_scenes length boolean array for whether heuristic is active 
        value_history: list of (step, value) tuples for each process
    Returns:

        """
    planner_inputs = [{} for e in range(num_scenes)]
    for e, p_input in enumerate(planner_inputs):
        p_input['goal'] = global_goals[e]
        p_input['map_pred'] = global_input[e, 0, :, :].detach().cpu().numpy()
        p_input['exp_pred'] = global_input[e, 1, :, :].detach().cpu().numpy()
        p_input['pose_pred'] = planner_pose_inputs[e]
        p_input['heuristic_active'] = heuristic[e]
        p_input['value_history'] = value_history[e] if value_history is not None else None
        if full_map is not None:
            p_input['full_map'] = full_map[e].detach().cpu().numpy()
        if local_map is not None:
            p_input['local_map'] = local_map[e].detach().cpu().numpy()

    # Output stores local goals as well as the the ground-truth action
    envs.visualize_map(planner_inputs)


class TensorboardLogger:
    def __init__(self, writer):
        self.writer = writer
        self.stats = {}

    def log(self, tag, value):
        if tag not in self.stats:
            self.stats[tag] = []
        self.stats[tag].append(value)

    def write(self, step):
        for tag, values in self.stats.items():
            if len(values) > 0:
                self.writer.add_scalar(tag, np.mean(values), step)
