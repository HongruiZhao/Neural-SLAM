import numpy as np

import env.utils.depth_utils as du
import matplotlib.pyplot as plt

class MapBuilder(object):
    def __init__(self, params):
        self.params = params
        self.debug = params['debug']
        frame_width = params['frame_width']
        frame_height = params['frame_height']
        fov = params['fov']
        self.camera_matrix = du.get_camera_matrix(
            frame_width,
            frame_height,
            fov)
        self.vision_range = params['vision_range']

        self.map_size_cm = params['map_size_cm']
        self.resolution = params['resolution'] # args.map_resolution
        agent_min_z = params['agent_min_z']
        agent_max_z = params['agent_max_z']
        self.z_bins = [agent_min_z, agent_max_z]
        self.du_scale = params['du_scale']
        self.visualize = params['visualize']
        self.obs_threshold = params['obs_threshold']

        self.map = np.zeros((self.map_size_cm // self.resolution,
                             self.map_size_cm // self.resolution,
                             len(self.z_bins) + 1), dtype=np.float32)

        self.agent_height = params['agent_height'] # in cm
        self.agent_view_angle = params['agent_view_angle']
        return

    def update_map(self, depth, current_pose):
        with np.errstate(invalid="ignore"):
            depth[depth > self.vision_range * self.resolution] = np.NaN
        point_cloud = du.get_point_cloud_from_z(depth, self.camera_matrix, \
                                                scale=self.du_scale)
        # visualization for debugging
        if self.debug:
            flat_PC = point_cloud.reshape(-1, 3) 
            plt.figure()
            plt.imshow(depth, cmap='viridis')
            plt.colorbar(label='Depth Value')
            plt.savefig('./debug/depth.png')

            fig = plt.figure()
            ax = fig.add_subplot(111, projection='3d')
            ax.scatter(flat_PC[:,0], flat_PC[:,1], flat_PC[:,2], s=1, c=flat_PC[:,1], cmap='plasma', alpha=0.8) # Colo
            ax.view_init(elev=20, azim=-60) # Elevation and azimuth angles
            plt.savefig('./debug/point_cloud.png')
        
        agent_view = du.transform_camera_view(point_cloud,
                                              self.agent_height, # in cm
                                              self.agent_view_angle)

        shift_loc = [self.vision_range * self.resolution // 2, 0, np.pi / 2.0]

        agent_view_centered = du.transform_pose(agent_view, shift_loc)

        # how many points in each bins. 
        # 3 z bins: x<25mm, 25mm<=x<150mm, 150mm<=x
        agent_view_flat = du.bin_points(
            agent_view_centered,
            self.vision_range, # map size = vision_range. thus 64x64 bins
            self.z_bins,
            self.resolution)

        # visualization for debugging
        if self.debug:
            plt.figure()
            for i in range(3):
                plt.subplot(1,3,i+1)
                plt.imshow(agent_view_flat[:,:,i], cmap='viridis')
            plt.savefig('./debug/agent_map.png')

        # only care about second bin 25mm<=x<150mm,
        agent_view_cropped = agent_view_flat[:, :, 1]
        # occupied or free if num of points > threshold
        agent_view_cropped = agent_view_cropped / self.obs_threshold
        agent_view_cropped[agent_view_cropped >= 0.5] = 1.0
        agent_view_cropped[agent_view_cropped < 0.5] = 0.0

        agent_view_explored = agent_view_flat.sum(2)
        agent_view_explored[agent_view_explored > 0] = 1.0

        if self.debug:
            plt.figure()
            plt.imshow(agent_view_explored, cmap='viridis')
            plt.savefig('./debug/agent_explored.png')

        geocentric_pc = du.transform_pose(agent_view, current_pose)

        geocentric_flat = du.bin_points(
            geocentric_pc,
            self.map.shape[0],
            self.z_bins,
            self.resolution) 

        self.map = self.map + geocentric_flat

        map_gt = self.map[:, :, 1] / self.obs_threshold
        map_gt[map_gt >= 0.5] = 1.0
        map_gt[map_gt < 0.5] = 0.0

        explored_gt = self.map.sum(2)
        explored_gt[explored_gt > 1] = 1.0

        if self.debug:
            plt.figure()
            plt.imshow(map_gt, cmap='viridis')
            plt.savefig('./debug/map_gt.png')
            plt.figure()
            plt.imshow(explored_gt, cmap='viridis')
            plt.savefig('./debug/explored_gt.png')

        return agent_view_cropped, map_gt, agent_view_explored, explored_gt

    def get_st_pose(self, current_loc):
        loc = [- (current_loc[0] / self.resolution
                  - self.map_size_cm // (self.resolution * 2)) / \
               (self.map_size_cm // (self.resolution * 2)),
               - (current_loc[1] / self.resolution
                  - self.map_size_cm // (self.resolution * 2)) / \
               (self.map_size_cm // (self.resolution * 2)),
               90 - np.rad2deg(current_loc[2])]
        return loc

    def reset_map(self, map_size):
        self.map_size_cm = map_size

        self.map = np.zeros((self.map_size_cm // self.resolution,
                             self.map_size_cm // self.resolution,
                             len(self.z_bins) + 1), dtype=np.float32)

    def get_map(self):
        return self.map
