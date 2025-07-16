import torch.nn.functional as F
import torch 
from torch.nn.utils import parameters_to_vector as p2v
import torch.optim as optim

import numpy as np
import random
import os
from scipy.spatial.transform import Rotation # to process quaternion 

# Local imports
from model.scene_rep import JointEncoding
from model.keyframe import KeyFrameDatabase
from optimization.utils import at_to_transform_matrix, qt_to_transform_matrix, matrix_to_axis_angle, matrix_to_quaternion
from utils import coordinates, extract_mesh




class Mapping():
    def __init__(self, config, id, dataset_info):
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.agent_id = id 
        self.dataset_info = dataset_info 

        self.create_bounds()
        self.create_pose_data()
        self.get_pose_representation()
        self.keyframeDatabase = self.create_kf_database(config)
        self.model = JointEncoding(config, self.bounding_box).to(self.device)
        self.fix_decoder = config['multi_agents']['fix_decoder']
        self.create_optimizer()

        self.total_loss = []
      

        
    def get_pose_representation(self):
        '''
        Get the pose representation axis-angle or quaternion
        '''
        if self.config['training']['rot_rep'] == 'axis_angle':
            self.matrix_to_tensor = matrix_to_axis_angle
            self.matrix_from_tensor = at_to_transform_matrix
            print('Using axis-angle as rotation representation, identity init would cause inf')
        
        elif self.config['training']['rot_rep'] == "quat":
            print("Using quaternion as rotation representation")
            self.matrix_to_tensor = matrix_to_quaternion
            self.matrix_from_tensor = qt_to_transform_matrix
        else:
            raise NotImplementedError
        

    def create_pose_data(self):
        '''
        Create the pose data
        '''
        self.est_c2w_data = {}
        self.est_c2w_data_rel = {}
    

    def create_bounds(self):
        '''
        Get the pre-defined bounds for the scene
        '''
        self.bounding_box = torch.from_numpy(np.array(self.config['mapping']['bound'])).to(torch.float32).to(self.device)
        self.marching_cube_bound = torch.from_numpy(np.array(self.config['mapping']['marching_cubes_bound'])).to(torch.float32).to(self.device)


    def create_kf_database(self, config):  
        '''
        Create the keyframe database
        '''
        num_kf = int(self.dataset_info['num_frames'] // self.config['mapping']['keyframe_every'] + 1)  
        print('#kf:', num_kf)
        print('#Pixels to save:', self.dataset_info['num_rays_to_save'])
        return KeyFrameDatabase(config, 
                                self.dataset_info['H'], 
                                self.dataset_info['W'], 
                                num_kf, 
                                self.dataset_info['num_rays_to_save'], 
                                self.device)
 

    def save_state_dict(self, save_path):
        torch.save(self.model.state_dict(), save_path)
    

    def load(self, load_path):
        self.model.load_state_dict(torch.load(load_path))


    def save_ckpt(self, save_path):
        '''
        Save the model parameters and the estimated pose
        '''
        save_dict = {'pose': self.est_c2w_data,
                     'pose_rel': self.est_c2w_data_rel,
                     'model': self.model.state_dict()}
        torch.save(save_dict, save_path)
        print('Save the checkpoint')


    def load_ckpt(self, load_path):
        '''
        Load the model parameters and the estimated pose
        '''
        dict = torch.load(load_path)
        self.model.load_state_dict(dict['model'])
        self.est_c2w_data = dict['pose']
        self.est_c2w_data_rel = dict['pose_rel']


    def select_samples(self, H, W, samples):
        '''
        randomly select samples from the image
        '''
        #indice = torch.randint(H*W, (samples,))
        indice = random.sample(range(H * W), int(samples))
        indice = torch.tensor(indice)
        return indice


    def get_loss_from_ret(self, ret, rgb=True, sdf=True, depth=True, fs=True, smooth=False):
        '''
        Get the training loss
        '''
        loss = 0
        if rgb:
            loss += self.config['training']['rgb_weight'] * ret['rgb_loss']
        if depth:
            loss += self.config['training']['depth_weight'] * ret['depth_loss']
        if sdf:
            loss += self.config['training']['sdf_weight'] * ret["sdf_loss"]
        if fs:
            loss +=  self.config['training']['fs_weight'] * ret["fs_loss"]
        
        if smooth and self.config['training']['smooth_weight']>0:
            loss += self.config['training']['smooth_weight'] * self.smoothness(self.config['training']['smooth_pts'], 
                                                                                  self.config['training']['smooth_vox'], 
                                                                                  margin=self.config['training']['smooth_margin'])
        
        return loss             


    def first_frame_mapping(self, batch, n_iters=100):
        '''
        First frame mapping
        Params:
            batch['c2w']: [1, 4, 4]
            batch['rgb']: [1, H, W, 3]
            batch['depth']: [1, H, W, 1]
            batch['direction']: [1, H, W, 3]
        Returns:
            ret: dict
            loss: float
        
        '''
        print(f'Agent {self.agent_id} First frame mapping...')
        c2w = batch['c2w'][0].to(self.device)
        self.est_c2w_data[0] = c2w
        self.est_c2w_data_rel[0] = c2w

        self.model.train()

        # Training
        for i in range(n_iters):
            self.map_optimizer.zero_grad()
            indice = self.select_samples(self.dataset_info['H'], self.dataset_info['W'], self.config['mapping']['sample'])
            
            indice_h, indice_w = indice % (self.dataset_info['H']), indice // (self.dataset_info['H'])
            rays_d_cam = batch['direction'].squeeze(0)[indice_h, indice_w, :].to(self.device)
            target_s = batch['rgb'].squeeze(0)[indice_h, indice_w, :].to(self.device)
            target_d = batch['depth'].squeeze(0)[indice_h, indice_w].to(self.device).unsqueeze(-1)

            rays_o = c2w[None, :3, -1].repeat(self.config['mapping']['sample'], 1)
            rays_d = torch.sum(rays_d_cam[..., None, :] * c2w[:3, :3], -1)

            # Forward
            ret = self.model.forward(rays_o, rays_d, target_s, target_d)
            loss = self.get_loss_from_ret(ret)
            loss.backward()

            self.map_optimizer.step()

        # First frame will always be a keyframe
        self.keyframeDatabase.add_keyframe(batch, filter_depth=self.config['mapping']['filter_depth'])
        if self.config['mapping']['first_mesh']:
            self.save_mesh(0)
        
        print(f'Agent {self.agent_id} First frame mapping done')
        return ret, loss


    def smoothness(self, sample_points=256, voxel_size=0.1, margin=0.05, color=False):
        '''
        Smoothness loss of feature grid
        '''
        volume = self.bounding_box[:, 1] - self.bounding_box[:, 0]

        grid_size = (sample_points-1) * voxel_size
        offset_max = self.bounding_box[:, 1]-self.bounding_box[:, 0] - grid_size - 2 * margin

        offset = torch.rand(3).to(offset_max) * offset_max + margin
        coords = coordinates(sample_points - 1, 'cpu', flatten=False).float().to(volume)
        pts = (coords + torch.rand((1,1,1,3)).to(volume)) * voxel_size + self.bounding_box[:, 0] + offset

        if self.config['grid']['tcnn_encoding']:
            pts_tcnn = (pts - self.bounding_box[:, 0]) / (self.bounding_box[:, 1] - self.bounding_box[:, 0])
        

        sdf = self.model.query_sdf(pts_tcnn, embed=True)
        tv_x = torch.pow(sdf[1:,...]-sdf[:-1,...], 2).sum()
        tv_y = torch.pow(sdf[:,1:,...]-sdf[:,:-1,...], 2).sum()
        tv_z = torch.pow(sdf[:,:,1:,...]-sdf[:,:,:-1,...], 2).sum()

        loss = (tv_x + tv_y + tv_z)/ (sample_points**3)

        return loss
               

    def global_BA(self, batch, cur_frame_id):
        '''
        Global bundle adjustment that includes all the keyframes and the current frame
        Params:
            batch['c2w']: ground truth camera pose [1, 4, 4]
            batch['rgb']: rgb image [1, H, W, 3]
            batch['depth']: depth image [1, H, W]
            batch['direction']: view direction [1, H, W, 3]
            cur_frame_id: current frame id
        '''

        # all the KF poses: 0, 5, 10, ...
        poses = torch.stack([self.est_c2w_data[i] for i in range(0, cur_frame_id, self.config['mapping']['keyframe_every'])])
        poses_fixed = torch.nn.parameter.Parameter(poses).to(self.device)
        current_pose = self.est_c2w_data[cur_frame_id][None,...]
        poses_all = torch.cat([poses_fixed, current_pose], dim=0)

        # Set up optimizer
        self.map_optimizer.zero_grad()
        
        current_rays = torch.cat([batch['direction'], batch['rgb'], batch['depth'][..., None]], dim=-1)
        current_rays = current_rays.reshape(-1, current_rays.shape[-1]) 

        mean_total_loss = 0
        for i in range(self.config['mapping']['iters']):

            # Sample rays with real frame ids
            # rays [bs, 7]
            # frame_ids [bs]
            rays, ids = self.keyframeDatabase.sample_global_rays(self.config['mapping']['sample'])

            #TODO: Checkpoint...
            idx_cur = random.sample(range(0, self.dataset_info['H'] * self.dataset_info['W']),max(self.config['mapping']['sample'] // len(self.keyframeDatabase.frame_ids), self.config['mapping']['min_pixels_cur']))
            current_rays_batch = current_rays[idx_cur, :]

            rays = torch.cat([rays, current_rays_batch], dim=0) # N, 7
            ids_all = torch.cat([ids//self.config['mapping']['keyframe_every'], -torch.ones((len(idx_cur)))]).to(torch.int64)


            rays_d_cam = rays[..., :3].to(self.device)
            target_s = rays[..., 3:6].to(self.device)
            target_d = rays[..., 6:7].to(self.device)

            # [N, Bs, 1, 3] * [N, 1, 3, 3] = (N, Bs, 3)
            rays_d = torch.sum(rays_d_cam[..., None, None, :] * poses_all[ids_all, None, :3, :3], -1)
            rays_o = poses_all[ids_all, None, :3, -1].repeat(1, rays_d.shape[1], 1).reshape(-1, 3)
            rays_d = rays_d.reshape(-1, 3)

            ret = self.model.forward(rays_o, rays_d, target_s, target_d)

            self.map_optimizer.zero_grad()
            loss = self.get_loss_from_ret(ret, smooth=True)
            loss.backward()
            self.map_optimizer.step()

            mean_total_loss += loss.item()


        # save loss info 
        mean_total_loss /= self.config['mapping']['iters']
        self.total_loss.append( mean_total_loss )



    def tracking_render(self, batch, frame_id):
        '''
            just save ground truth pose
        '''
        c2w_gt = batch['c2w'][0].to(self.device)
        self.est_c2w_data[frame_id] = c2w_gt


    def create_optimizer(self):
        '''
        Create optimizer for mapping
        '''

        trainable_parameters = [{'params': self.model.decoder.parameters(), 'weight_decay': 1e-6, 'lr': self.config['mapping']['lr_decoder']},
                                    {'params': self.model.embed_fn.parameters(), 'eps': 1e-15, 'lr': self.config['mapping']['lr_embed']}]

        if not self.config['grid']['oneGrid']:
            trainable_parameters.append({'params': self.model.embed_fn_color.parameters(), 'eps': 1e-15, 'lr': self.config['mapping']['lr_embed_color']})
        
        self.map_optimizer = optim.Adam(trainable_parameters, betas=(0.9, 0.99))
        
    
    def save_mesh(self, i, voxel_size=0.05):
        mesh_savepath = os.path.join(self.config['data']['output'], self.config['data']['exp_name'], f'agent_{self.agent_id}', 'mesh_track{}.ply'.format(i))
        if self.config['mesh']['render_color']:
            color_func = self.model.render_surface_color
        else:
            color_func = self.model.query_color
        extract_mesh(self.model.query_sdf, 
                        self.config, 
                        self.bounding_box, 
                        color_func=color_func, 
                        marching_cube_bound=self.marching_cube_bound, 
                        voxel_size=voxel_size, 
                        mesh_savepath=mesh_savepath)    


    def run(self, i, batch):
        """
            @param i: current step
            @param batch:
        """
        # First frame mapping
        if i == 0:
            self.first_frame_mapping(batch, self.config['mapping']['first_iters'])
            return 
        
        # Tracking + Mapping
        self.tracking_render(batch, i)

        if i%self.config['mapping']['map_every']==0:
            self.global_BA(batch, i)
            
        # Add keyframe
        if i % self.config['mapping']['keyframe_every'] == 0:
            self.keyframeDatabase.add_keyframe(batch, filter_depth=self.config['mapping']['filter_depth'])
            #print(f'\nAgent {self.agent_id} add keyframe:{i}')

        if i % self.config['mesh']['vis']==0:
            model_savepath = os.path.join(self.config['data']['output'], self.config['data']['exp_name'], f'agent_{self.agent_id}', 'checkpoint_{}.pt'.format(i)) 
            self.save_ckpt(model_savepath)
            self.save_mesh(i, voxel_size=self.config['mesh']['voxel_eval'])



def get_camera_rays(H, W, fx, fy=None, cx=None, cy=None, type='OpenGL'):
    """Get ray origins, directions from a pinhole camera."""
    #  ----> i
    # |
    # |
    # X
    # j
    i, j = torch.meshgrid(torch.arange(W, dtype=torch.float32),
                       torch.arange(H, dtype=torch.float32), indexing='xy')
    
    # View direction (X, Y, Lambda) / lambda
    # Move to the center of the screen
    #  -------------
    # |      y      |
    # |      |      |
    # |      .-- x  |
    # |             |
    # |             |
    #  -------------

    if cx is None:
        cx, cy = 0.5 * W, 0.5 * H

    if fy is None:
        fy = fx
    if type ==  'OpenGL':
        dirs = torch.stack([(i - cx)/fx, -(j - cy)/fy, -torch.ones_like(i)], -1)
    elif type == 'OpenCV':
        dirs = torch.stack([(i - cx)/fx, (j - cy)/fy, torch.ones_like(i)], -1)
    else:
        raise NotImplementedError()

    rays_d = dirs
    return rays_d




def data_loading(rgb_image, depth_image, agent_pos, agent_q, step, rays_d):
    """
        @param rgb_image: (H,W,4) RGBA
        @return batch
    """
    
    rgb_image = torch.from_numpy(rgb_image.astype(np.float32) / 255.)[:,:,:3] # Normalize to [0, 1]
    depth_image = torch.from_numpy(depth_image.astype(np.float32)) # in m

    # Extract rotation (orientation)
    rotation = Rotation.from_quat([
        agent_q.x,
        agent_q.y,
        agent_q.z,
        agent_q.w
    ])
    # Convert rotation to rotation matrix
    rotation_matrix = rotation.as_matrix()
    # Create the 4x4 transformation matrix
    transformation_matrix = np.eye(4)  # Initialize as identity matrix
    transformation_matrix[:3, :3] = rotation_matrix  # Set rotation part
    transformation_matrix[:3, 3] = agent_pos  # Set translation part
    pose  = torch.from_numpy(transformation_matrix.astype(np.float32))
    

    # Create a dummy batch
    batch = {
        'frame_id': step,
        'c2w': pose.unsqueeze(0),  
        'rgb': rgb_image.unsqueeze(0),  # [1, H, W, 3]
        'depth': depth_image.unsqueeze(0),  # [1, H, W]
        'direction': rays_d.unsqueeze(0)
    }

    return batch