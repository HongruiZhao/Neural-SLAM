import numpy as np
import torch
import torch.nn as nn


class MappingHandler:
    """
    Class handling the initialization and update of maps and poses.
    """
    def __init__(self, args, num_scenes: int, device: torch.device):
        """
        Initialize the MappingHandler.
        Args:
            args: Command line arguments.
            num_scenes: Number of parallel environments.
            device: Torch device.
        """
        self.args = args
        self.num_scenes = num_scenes
        self.device = device
        
        # Map variables initialization
        self.map_size = args.map_size_cm // args.map_resolution
        self.full_w, self.full_h = self.map_size, self.map_size
        self.local_w, self.local_h = int(self.full_w / args.global_downscaling), \
                                     int(self.full_h / args.global_downscaling)

        self.full_map = torch.zeros(num_scenes, 4, self.full_w, self.full_h).float().to(device)
        self.local_map = torch.zeros(num_scenes, 4, self.local_w, self.local_h).float().to(device)

        # Initial full and local pose. pose (x, y, o)
        # x,y: xy coordinate in meters 
        # o: orientation (clockwise from x) in deg
        self.full_pose = torch.zeros(num_scenes, 3).float().to(device)
        self.local_pose = torch.zeros(num_scenes, 3).float().to(device)

        self.origins = np.zeros((num_scenes, 3))
        self.lmb = np.zeros((num_scenes, 4)).astype(int)

        ### Planner pose inputs has 7 dimensions
        ### 1-3 store continuous global agent location
        ### 4-7 store local map boundaries
        self.planner_pose_inputs = np.zeros((num_scenes, 7))


    def _get_local_map_boundaries(self, agent_loc):
        loc_r, loc_c = agent_loc
        local_w, local_h = self.local_w, self.local_h
        full_w, full_h = self.full_w, self.full_h
        global_downscaling = self.args.global_downscaling

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


    def _init_map_and_pose(self):
        self.full_map.fill_(0.)
        self.full_pose.fill_(0.)
        self.full_pose[:, :2] = self.args.map_size_cm / 100.0 / 2.0 # initially at the map center

        locs = self.full_pose.cpu().numpy()
        self.planner_pose_inputs[:, :3] = locs
        for e in range(self.num_scenes):
            r, c = locs[e, 1], locs[e, 0]
            loc_r, loc_c = [int(r * 100.0 / self.args.map_resolution),
                            int(c * 100.0 / self.args.map_resolution)]

            self.full_map[e, 2:, loc_r - 1:loc_r + 2, loc_c - 1:loc_c + 2] = 1.0

            self.lmb[e] = self._get_local_map_boundaries((loc_r, loc_c))

            self.planner_pose_inputs[e, 3:] = self.lmb[e]
            self.origins[e] = [self.lmb[e][0] * self.args.map_resolution / 100.0,
                            self.lmb[e][2] * self.args.map_resolution / 100.0, 0.]  # cm to m

        for e in range(self.num_scenes):
            self.local_map[e] = self.full_map[e, :, self.lmb[e, 2]:self.lmb[e, 3], self.lmb[e, 0]:self.lmb[e, 1]]
            self.local_pose[e] = self.full_pose[e] - \
                            torch.from_numpy(self.origins[e]).to(self.device).float()


    def get_full_map_pose(self, infos, is_global_step=False):
        """get maps from habitat sim for global policy input
        Args:
            infos: outputs from sim envs, including maps and gt pose.
            is_global_step: whether to update lmb and origins
        """
        def extract_map(e, key):
            val = infos[e][key]
            if isinstance(val, tuple):
                return val[0]
            return val

        all_maps = np.stack([extract_map(e, 'map') for e in range(self.num_scenes)])
        self.full_map[:, 0, :, :] = torch.from_numpy(all_maps).to(self.device)

        if self.args.global_arch == 'lena':
            all_uncert_maps = np.stack([extract_map(e, 'uncert_map') for e in range(self.num_scenes)])
            self.full_map[:, 1, :, :] = torch.from_numpy(all_uncert_maps).to(self.device)
        else:
            all_explored_maps = np.stack([extract_map(e, 'explored_map') for e in range(self.num_scenes)])
            self.full_map[:, 1, :, :] = torch.from_numpy(all_explored_maps).to(self.device)

        # Update full_pose
        self.full_pose = torch.from_numpy(np.asarray(
            [infos[e]['gt_pose'] for e in range(self.num_scenes)])).float().to(self.device)
        # Normalize orientation
        self.full_pose[:, 2] = self.full_pose[:, 2] % 360
        self.full_pose[:, 2] = self.full_pose[:, 2] - 360 * (self.full_pose[:, 2] > 180)

        # Update 3rd and 4th channels
        self.full_map[:, 2, :, :].fill_(0.)
        locs = self.full_pose.cpu().numpy()
        for e in range(self.num_scenes):
            r, c = locs[e, 1], locs[e, 0]
            loc_r, loc_c = [int(r * 100.0 / self.args.map_resolution),
                            int(c * 100.0 / self.args.map_resolution)]
            
            self.full_map[e, 2, loc_r - 2:loc_r + 3, loc_c - 2:loc_c + 3] = 1.
            self.full_map[e, 3, loc_r - 2:loc_r + 3, loc_c - 2:loc_c + 3] = 1.

        if is_global_step:
            for e in range(self.num_scenes):
                r, c = locs[e, 1], locs[e, 0]
                loc_r, loc_c = [int(r * 100.0 / self.args.map_resolution),
                                int(c * 100.0 / self.args.map_resolution)]
                self.lmb[e] = self._get_local_map_boundaries((loc_r, loc_c))
                self.planner_pose_inputs[e, 3:] = self.lmb[e]
                self.origins[e] = [self.lmb[e][0] * self.args.map_resolution / 100.0,
                                   self.lmb[e][2] * self.args.map_resolution / 100.0, 0.]


    def get_local_map_pose(self):
        """
        Update self.local_map and self.local_pose.
        """
        for e in range(self.num_scenes):
            self.local_map[e] = self.full_map[e, :, self.lmb[e, 2]:self.lmb[e, 3], self.lmb[e, 0]:self.lmb[e, 1]]
            self.local_pose[e] = self.full_pose[e] - \
                                 torch.from_numpy(self.origins[e]).to(self.device).float()
        
        # Update planner_pose_inputs
        self.planner_pose_inputs[:, :3] = self.local_pose.cpu().numpy() + self.origins


    def get_global_input(self):
        """
        Prepare global policy input.
        Returns:
            Tuple of global input tensor and orientation tensor.
        """
        global_input = torch.zeros(self.num_scenes, 8, self.local_w, self.local_h).to(self.device)
        global_input[:, 0:4, :, :] = self.local_map.detach()
        global_input[:, 4:, :, :] = nn.MaxPool2d(self.args.global_downscaling)(self.full_map)
        
        global_orientation = torch.zeros(self.num_scenes, 1).long().to(self.device)
        locs = self.full_pose.cpu().numpy()
        for e in range(self.num_scenes):
            global_orientation[e] = np.clip(int((locs[e, 2] + 180.0) / 5.), 0, 71)
            
        return global_input, global_orientation
