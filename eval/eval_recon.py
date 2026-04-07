import argparse
import os
import random
import time

import numpy as np
import open3d as o3d
import torch
import trimesh
from scipy.spatial import cKDTree as KDTree
from tqdm import trange
import quaternion
import matplotlib.pyplot as plt
from pathlib import Path

'''
reconstruction evaluation tools
modified from https://github.com/cvg/nice-slam/blob/master/src/tools/eval_recon.py
'''


def normalize(x):
    return x / np.linalg.norm(x)


def viewmatrix(z, up, pos):
    vec2 = normalize(z)
    vec1_avg = up
    vec0 = normalize(np.cross(vec1_avg, vec2))
    vec1 = normalize(np.cross(vec2, vec0))
    m = np.stack([vec0, vec1, vec2, pos], 1)
    return m


def completion_ratio(gt_points, rec_points, dist_th=0.05):
    gen_points_kd_tree = KDTree(rec_points)
    distances, _ = gen_points_kd_tree.query(gt_points)
    comp_ratio = np.mean((distances < dist_th).astype(np.float32))
    return comp_ratio, distances


def accuracy(gt_points, rec_points):
    gt_points_kd_tree = KDTree(gt_points)
    distances, _ = gt_points_kd_tree.query(rec_points)
    acc = np.mean(distances)
    return acc


def completion(gt_points, rec_points):
    gt_points_kd_tree = KDTree(rec_points)
    distances, _ = gt_points_kd_tree.query(gt_points)
    comp = np.mean(distances)
    return comp


def get_align_transformation(initial_state, map_center=12.0):
    """
    Get the transformation matrix to align the reconstructed mesh to the ground truth mesh.
    """
    #  flip axes
    trans_habitat2mesh = np.array([ [1,0,0,0],
                                    [0,0,-1,0],
                                    [0,1,0,0],
                                    [0,0,0,1]])
    trans_step1 = trans_habitat2mesh.copy()

    # translate to center at robot frame
    # map_center = 12.0 # in m
    rec_ori2robot = np.array([-map_center, 0, -map_center])
    trans_step1[:3,3] = -trans_habitat2mesh[:3,:3]@rec_ori2robot

    # rotation around robot frame to account for initial orientation
    eulers = trimesh.transformations.euler_from_quaternion(initial_state['rotation'], axes='sxyz')
    trans_step2 = trimesh.transformations.euler_matrix(eulers[0], -eulers[2], eulers[1], 'sxyz') # axes flip 
    # TODO: WTF?
    #trans_step2[:3,:3] = trans_step2[:3,:3].T # required for Cantwell
    trans_step2[:3,:3] = trans_step2[:3,:3] # for Eudora, Annawan

    # translation to account for initial position
    agent_height = 1.25 # in m
    habitat_ori2robot = initial_state['position'].copy()
    habitat_ori2robot[1] += agent_height   
    habitat_ori2robot = trans_habitat2mesh[:3,:3] @ habitat_ori2robot
    trans_step2[:3,3] = habitat_ori2robot

    trans_combined = trans_step2 @ trans_step1
    return trans_combined


def check_proj(points, W, H, fx, fy, cx, cy, c2w):
    """
    Check if points can be projected into the camera view.
    """
    c2w = c2w.copy()
    c2w[:3, 1] *= -1.0
    c2w[:3, 2] *= -1.0
    points = torch.from_numpy(points).cuda().clone()
    w2c = np.linalg.inv(c2w)
    w2c = torch.from_numpy(w2c).cuda().float()
    K = torch.from_numpy(
        np.array([[fx, .0, cx], [.0, fy, cy], [.0, .0, 1.0]]).reshape(3, 3)).cuda()
    ones = torch.ones_like(points[:, 0]).reshape(-1, 1).cuda()
    homo_points = torch.cat(
        [points, ones], dim=1).reshape(-1, 4, 1).cuda().float()  # (N, 4)
    cam_cord_homo = w2c@homo_points  # (N, 4, 1)=(4,4)*(N, 4, 1)
    cam_cord = cam_cord_homo[:, :3]  # (N, 3, 1)
    cam_cord[:, 0] *= -1
    uv = K.float()@cam_cord.float()
    z = uv[:, -1:] + 1e-5
    uv = uv[:, :2]/z
    uv = uv.float().squeeze(-1).cpu().numpy()
    edge = 0
    mask = (0 <= -z[:, 0, 0].cpu().numpy()) & (uv[:, 0] < W -
                                               edge) & (uv[:, 0] > edge) & (uv[:, 1] < H-edge) & (uv[:, 1] > edge)
    return mask.sum() > 0


def calc_3d_metric(rec_meshfile, gt_meshfile, initial_state, align=True):
    """
    3D reconstruction metric.
    """
    mesh_rec = trimesh.load(rec_meshfile, process=False)
    mesh_gt = trimesh.load(gt_meshfile, process=False, force='mesh') # force: otherwise.glb are loaded as scene

    if align:
        trans_rec_to_gt = get_align_transformation(initial_state)
        mesh_gt.apply_transform(np.linalg.inv(trans_rec_to_gt)) 

    rec_pc = trimesh.sample.sample_surface(mesh_rec, 200000)
    rec_pc_tri = trimesh.PointCloud(vertices=rec_pc[0])

    gt_pc = trimesh.sample.sample_surface(mesh_gt, 200000)
    gt_pc_tri = trimesh.PointCloud(vertices=gt_pc[0])
    
    # Precision 5cm
    precision_rec, rec_dists = completion_ratio(rec_pc_tri.vertices, gt_pc_tri.vertices, dist_th=0.05)
    # Recall 5cm
    recall_rec, gt_dists = completion_ratio(gt_pc_tri.vertices, rec_pc_tri.vertices, dist_th=0.05)
    
    precision_rec *= 100
    recall_rec *= 100
    f1_rec = 2 * precision_rec * recall_rec / (precision_rec + recall_rec + 1e-6)

    # Visualize metrics
    exp_name = Path(rec_meshfile).parts[-3]
    mesh_name = Path(rec_meshfile).stem
    visualize_metrics(rec_pc_tri.vertices, gt_pc_tri.vertices, exp_name, mesh_name, rec_dists=rec_dists, gt_dists=gt_dists)

    print('precision: {:.2f}'.format(precision_rec) )
    print('recall: {:.2f}'.format(recall_rec) )
    print('f1: {:.2f}'.format(f1_rec) )

    return {'Precision 5cm': precision_rec, 'Recall 5cm': recall_rec, 'F1 5cm': f1_rec}


def visualize_metrics(rec_pc, gt_pc, exp_name, mesh_name, 
                      rec_dists=None, gt_dists=None, 
                      grid_res=200, map_bound_x=[-24,0], map_bound_z=[-24,0]):
    """
    Visualize precision, recall, and F1 score on a 2D grid.
    """
    # Create grids
    precision_grid = np.zeros((grid_res, grid_res))
    recall_grid = np.zeros((grid_res, grid_res))
    f1_grid = np.zeros((grid_res, grid_res))

    rec_pts_per_cell = np.zeros((grid_res, grid_res))
    gt_pts_per_cell = np.zeros((grid_res, grid_res))
    
    rec_tp_per_cell = np.zeros((grid_res, grid_res))
    gt_tp_per_cell = np.zeros((grid_res, grid_res))

    # Discretize points into grid cells
    x_coords = np.clip(np.floor((rec_pc[:, 0] - map_bound_x[0]) / (map_bound_x[1] - map_bound_x[0]) * grid_res), 0, grid_res - 1).astype(int)
    z_coords = np.clip(np.floor((rec_pc[:, 2] - map_bound_z[0]) / (map_bound_z[1] - map_bound_z[0]) * grid_res), 0, grid_res - 1).astype(int)
    
    for i in range(len(x_coords)):
        rec_pts_per_cell[z_coords[i], x_coords[i]] += 1
    
    gt_x_coords = np.clip(np.floor((gt_pc[:, 0] - map_bound_x[0]) / (map_bound_x[1] - map_bound_x[0]) * grid_res), 0, grid_res - 1).astype(int)
    gt_z_coords = np.clip(np.floor((gt_pc[:, 2] - map_bound_z[0]) / (map_bound_z[1] - map_bound_z[0]) * grid_res), 0, grid_res - 1).astype(int)

    for i in range(len(gt_x_coords)):
        gt_pts_per_cell[gt_z_coords[i], gt_x_coords[i]] += 1

    # Precision
    if rec_dists is None:
        gt_kdtree = KDTree(gt_pc)
        rec_dists, _ = gt_kdtree.query(rec_pc)
        
    for i, dist in enumerate(rec_dists):
        if dist < 0.05:
            rec_tp_per_cell[z_coords[i], x_coords[i]] += 1

    # Recall
    if gt_dists is None:
        rec_kdtree = KDTree(rec_pc)
        gt_dists, _ = rec_kdtree.query(gt_pc)
        
    for i, dist in enumerate(gt_dists):
        if dist < 0.05:
            gt_tp_per_cell[gt_z_coords[i], gt_x_coords[i]] += 1
            
    # Calculate metrics per cell
    precision_grid = np.divide(rec_tp_per_cell, rec_pts_per_cell, out=np.zeros_like(rec_tp_per_cell), where=rec_pts_per_cell!=0)
    recall_grid = np.divide(gt_tp_per_cell, gt_pts_per_cell, out=np.zeros_like(gt_tp_per_cell), where=gt_pts_per_cell!=0)
    
    # F1 score
    f1_grid = np.divide(2 * precision_grid * recall_grid, precision_grid + recall_grid, out=np.zeros_like(precision_grid), where=(precision_grid + recall_grid)!=0)

    # Plotting
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    im1 = axes[0].imshow(precision_grid, cmap='viridis')
    axes[0].set_title('Precision 5cm')
    fig.colorbar(im1, ax=axes[0])

    im2 = axes[1].imshow(recall_grid, cmap='viridis')
    axes[1].set_title('Recall 5cm')
    fig.colorbar(im2, ax=axes[1])

    im3 = axes[2].imshow(f1_grid, cmap='viridis')
    axes[2].set_title('F1 5cm')
    fig.colorbar(im3, ax=axes[2])
    
    # Create folder and save plot
    output_dir = Path('eval_vis_results') / exp_name
    output_dir.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_dir / f'{mesh_name}.png')
    plt.close(fig)


def get_cam_position(gt_meshfile, sx=0.3, sy=0.6, sz=0.6, dx=0.0, dy=0.0, dz=0.0):
    mesh_gt = trimesh.load(gt_meshfile)
    # Tbw: world_to_bound, bound is defined at the centre of cuboid
    to_origin, extents = trimesh.bounds.oriented_bounds(mesh_gt)
    extents[2] *= sz
    extents[1] *= sy
    extents[0] *= sx
    # Twb: bound_to_world
    transform = np.linalg.inv(to_origin)
    transform[0, 3] += dx
    transform[1, 3] += dy
    transform[2, 3] += dz
    return extents, transform

#------------------------------------------------------
def render_depth_offscreen(mesh, width, height, fx, fy, cx, cy, c2w):
    """
    use OffscreenRenderer to render depth map of a mesh
    """
    renderer = o3d.visualization.rendering.OffscreenRenderer(width, height)
    
    # set the background color
    material = o3d.visualization.rendering.MaterialRecord()
    material.shader = "defaultLit"
    
    renderer.scene.add_geometry("mesh", mesh, material)
    
    # set the camera intrinsic parameters
    renderer.scene.camera.set_projection(
        fx, fy, cx, cy, 0.1, 20.0, width, height
    )
    
    # set the camera extrinsic parameters
    w2c = np.linalg.inv(c2w)
    renderer.scene.camera.look_at([0, 0, 0], [0, 0, 1], [0, 1, 0])  
    renderer.scene.camera.set_model_matrix(np.linalg.inv(w2c))  
    
    # render the depth image
    depth_image = renderer.render_to_depth_image()
    depth_array = np.asarray(depth_image)
    
    return depth_array



def calc_2d_metric(rec_meshfile, gt_meshfile, unseen_gt_pcd_file,
                   pose_file=None, gt_depth_render_file=None,
                   depth_render_file=None, suffix="virt_cams", align=True,
                   n_imgs=1000, not_counting_missing_depth=True,
                   sx=0.3, sy=0.6, sz=0.6, dx=0.0, dy=0.0, dz=0.0):
    """
    2D reconstruction metric, depth L1 loss. modified from NICE-SLAM
    use OffscreenRenderer to render depth maps
    """
    H = 500
    W = 500
    focal = 300
    fx = focal
    fy = focal
    cx = H/2.0-0.5
    cy = W/2.0-0.5

    gt_mesh = o3d.io.read_triangle_mesh(gt_meshfile)
    rec_mesh = o3d.io.read_triangle_mesh(rec_meshfile)
    pc_unseen = np.load(unseen_gt_pcd_file)

    if pose_file and os.path.exists(pose_file):
        sampled_poses = np.load(pose_file)["poses"]
        assert len(sampled_poses) == n_imgs
        print("Found saved rendering poses! Loading from disk!!!")
    else:
        sampled_poses = None
        print("Saved rendering poses NOT FOUND! Will do the sampling")
    if gt_depth_render_file and os.path.exists(gt_depth_render_file):
        gt_depth_renderings = np.load(gt_depth_render_file)["depths"]
        assert len(gt_depth_renderings) == n_imgs
        print("Found saved rendered gt depths! Loading from disk!!!")
    else:
        gt_depth_renderings = None
        print("Saved rendered gt depths NOT FOUND! Will re-render!!!")
    if depth_render_file and os.path.exists(depth_render_file):
        depth_renderings = np.load(depth_render_file)["depths"]
        assert len(depth_renderings) == n_imgs
        print("Found saved rendered reconstructed depth! Loading from disk!!!")
    else:
        depth_renderings = None
        print("Saved rendered reconstructed depth NOT FOUND! Will re-render!!!")

    gt_dir = os.path.dirname(unseen_gt_pcd_file)
    log_dir = os.path.dirname(rec_meshfile)

    if align:
        transformation = get_align_transformation(rec_meshfile, gt_meshfile)
        rec_mesh = rec_mesh.transform(transformation)

    # get vacant area inside the room
    extents, transform = get_cam_position(gt_meshfile, sx=sx, sy=sy, sz=sz, dx=dx, dy=dy, dz=dz)

    errors = []
    poses = []
    gt_depths = []
    depths = []
    
    for i in trange(n_imgs, smoothing=0):
        if sampled_poses is None:
            while True:
                # sample view, and check if unseen region is not inside the camera view
                # if inside, then needs to resample
                # camera-up (Y-direction) vector under world
                up = [0, 0, -1]
                # camera origin coord under world coordinate-frame, sampled within extents of the oriented bound
                origin = trimesh.sample.volume_rectangular(extents, 1, transform=transform)
                origin = origin.reshape(-1)
                # sampled target coord under world [tx, ty, tz]
                tx = round(random.uniform(-10000, +10000), 2)
                ty = round(random.uniform(-10000, +10000), 2)
                tz = round(random.uniform(-10000, +10000), 2)
                target = [tx, ty, tz]
                # look_at vector (camera-Z), from origin to target
                target = np.array(target)-np.array(origin)
                c2w = viewmatrix(target, up, origin)
                tmp = np.eye(4)
                tmp[:3, :] = c2w
                c2w = tmp
                seen = check_proj(pc_unseen, W, H, fx, fy, cx, cy, c2w)
                if (~seen):
                    break
            poses.append(c2w)
        else:
            c2w = sampled_poses[i]

        # use OffscreenRenderer to render depth maps
        if gt_depth_renderings is None:
            gt_depth = render_depth_offscreen(gt_mesh, W, H, fx, fy, cx, cy, c2w)
            gt_depths.append(gt_depth)
        else:
            gt_depth = gt_depth_renderings[i]
        
        if depth_renderings is None:
            ours_depth = render_depth_offscreen(rec_mesh, W, H, fx, fy, cx, cy, c2w)
            depths.append(ours_depth)
        else:
            ours_depth = depth_renderings[i]

        if not_counting_missing_depth:
            valid_mask = (gt_depth > 0.) & (gt_depth < 19.)
            if np.count_nonzero(valid_mask) <= 100:
                continue
            errors += [np.abs(gt_depth[valid_mask] - ours_depth[valid_mask]).mean()]
        else:
            errors += [np.abs(gt_depth-ours_depth).mean()]

    if pose_file is None:
        np.savez(os.path.join(gt_dir, "sampled_poses_{}.npz".format(n_imgs)), poses=poses)
    elif not os.path.exists(pose_file):
        np.savez(pose_file, poses=poses)

    if gt_depth_render_file is None:
        np.savez(os.path.join(gt_dir, "gt_depths_{}.npz".format(n_imgs)), depths=gt_depths)
    elif not os.path.exists(gt_depth_render_file):
        np.savez(gt_depth_render_file, depths=gt_depths)

    if depth_render_file is None:
        np.savez(os.path.join(log_dir, "depths_{}_{}.npz".format(suffix, n_imgs)), depths=depths)
    elif not os.path.exists(depth_render_file):
        np.savez(depth_render_file, depths=depths)

    errors = np.array(errors)
    # from m to cm
    print('Depth L1: ', errors.mean() * 100)
    return {"Depth L1": errors.mean() * 100}

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Arguments to evaluate the reconstruction."
    )
    parser.add_argument("--rec_mesh", type=str,
                        help="reconstructed mesh file path")
    parser.add_argument("--gt_mesh", type=str,
                        help="ground truth mesh file path")
    parser.add_argument("--ckpt_path", type=str, help="path to checkpoint file")
    args = parser.parse_args()


    assert args.ckpt_path is not None and os.path.exists(args.ckpt_path), "Please ensure you provided ckpt path and it exists!!!"
    initial_state = torch.load(args.ckpt_path, weights_only=False)["initial_state"]
    calc_3d_metric(args.rec_mesh, args.gt_mesh, 
                   initial_state={'position':initial_state['position'], 
                                  'rotation':quaternion.as_float_array(initial_state['rotation'])} 
                    )
