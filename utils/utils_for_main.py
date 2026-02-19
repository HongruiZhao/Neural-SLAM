import numpy as np
import torch 
            

def get_local_map_boundaries(agent_loc, local_sizes, full_sizes, global_downscaling):
    loc_r, loc_c = agent_loc
    local_w, local_h = local_sizes
    full_w, full_h = full_sizes

    if global_downscaling > 1:
        gx1, gy1 = loc_c - local_w // 2, loc_r - local_h // 2
        gx2, gy2 = gx1 + local_w, gy1 + local_h
        if gx1 < 0:
            gx1, gx2 = 0, local_w
        if gx2 > full_w:
            gx1, gx2 = full_w - local_w, full_w

        if gy1 < 0:
            gy1, gy2 = 0, local_h
        if gy2 > full_h:
            gy1, gy2 = full_h - local_h, full_h
    else:
        gx1, gx2, gy1, gy2 = 0, full_w, 0, full_h

    return [gx1, gx2, gy1, gy2]


def init_map_and_pose(full_map, full_pose, local_map, local_pose, planner_pose_inputs, origins, lmb,
                      local_w, local_h, full_w, full_h, 
                      map_size_cm, map_resolution, global_downscaling,
                      num_scenes, device):
    full_map.fill_(0.)
    full_pose.fill_(0.)
    full_pose[:, :2] = map_size_cm / 100.0 / 2.0 # initially at the map center

    locs = full_pose.cpu().numpy()
    planner_pose_inputs[:, :3] = locs
    for e in range(num_scenes):
        r, c = locs[e, 1], locs[e, 0]
        loc_r, loc_c = [int(r * 100.0 / map_resolution),
                        int(c * 100.0 / map_resolution)]

        full_map[e, 2:, loc_r - 1:loc_r + 2, loc_c - 1:loc_c + 2] = 1.0

        lmb[e] = get_local_map_boundaries(  (loc_r, loc_c),
                                            (local_w, local_h),   
                                            (full_w, full_h),
                                            global_downscaling)

        planner_pose_inputs[e, 3:] = lmb[e]
        origins[e] = [lmb[e][0] * map_resolution / 100.0,
                        lmb[e][2] * map_resolution / 100.0, 0.]  # cm to m

    for e in range(num_scenes):
        local_map[e] = full_map[e, :, lmb[e, 2]:lmb[e, 3], lmb[e, 0]:lmb[e, 1]]
        local_pose[e] = full_pose[e] - \
                        torch.from_numpy(origins[e]).to(device).float()
        
    return full_map, full_pose, local_map, local_pose, planner_pose_inputs, lmb


def get_map_from_envs(infos, local_map, local_pose, lmb, origins, num_scenes, global_arch, device):
        """get maps from habiat sim for global policy input
        Args:
            infos: outputs from sim envs, including maps and gt pose.
            local_map: num_scenes x 4 x local_w x local_h. cropped from full maps. x for col and y for row.
            local_pose: num_scenes x 3. x(m), y(m), o(deg) w.r.t origin
            lmb: local map boundaries. num_scenes x 4. (gx1, gx2, gy1, gy2). in pixels
            origins: (gx1, gy1) but in meters
            num_scenes:
            global_arch: 'lena' or 'NeuralSLAM'
            device:
        Returns:
            local_map: 
            local_pose: num_scenes x 3. x(m), y(m), o(deg) w.r.t origin
        """
        try:
            all_maps = np.stack([infos[e]['map'][lmb[e, 2]:lmb[e, 3], lmb[e, 0]:lmb[e, 1]] for e in range(num_scenes)]) 
            all_explored_maps = np.stack([infos[e]['explored_map'][lmb[e, 2]:lmb[e, 3], lmb[e, 0]:lmb[e, 1]] for e in range(num_scenes)])
        except TypeError: # [0]: info returns a tuple (map,) for some reasons
            all_maps = np.stack([infos[e]['map'][0][lmb[e, 2]:lmb[e, 3], lmb[e, 0]:lmb[e, 1]] for e in range(num_scenes)]) 
            all_explored_maps = np.stack([infos[e]['explored_map'][0][lmb[e, 2]:lmb[e, 3], lmb[e, 0]:lmb[e, 1]] for e in range(num_scenes)])
        torch_maps = torch.from_numpy(all_maps).to(device)
        torch_explored_maps = torch.from_numpy(all_explored_maps).to(device)
        local_map[:, 0, :, :] = torch_maps
        if global_arch == 'lena':
            all_uncert_maps = np.stack([infos[e]['uncert_map'][lmb[e, 2]:lmb[e, 3], lmb[e, 0]:lmb[e, 1]] for e in range(num_scenes)])
            torch_uncert_maps = torch.from_numpy(all_uncert_maps).to(device)
            local_map[:, 1, :, :] = torch_uncert_maps
        else:  
            local_map[:, 1, :, :] = torch_explored_maps
        local_pose = torch.from_numpy(np.asarray(
            [infos[env_idx]['gt_pose'] for env_idx
            in range(num_scenes)])).float().to(device) - \
            torch.from_numpy(origins).to(device).float()    
        # convert angle to be between -180 and 180
        local_pose[:,2] = local_pose[:,2] % 360
        local_pose[:,2] = local_pose[:,2] - 360*(local_pose[:,2] > 180)

        return local_map, local_pose


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
                     envs, heuristic=None, value_history=None):
    """for visualization
    Args:
        num_scenes: 
        global_goals: num_scenes x global waypoints. in pixels (scaled by map_resolution), local window origin.
        global_input: num_scenes x 8 x local_w x local_h maps. 
        planner_pose_inputs: 
            - num_scenes x 7. 
            - 1-3 store continuous global agent x(m), y(m), o (deg). 4-7 store local map boundaries (gx1, gx2, gy1, gy2) 
        envs: 
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
