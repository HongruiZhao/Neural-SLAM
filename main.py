import time
from collections import deque

import os

os.environ["OMP_NUM_THREADS"] = "1"
import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F

import gym
import logging
import json
from arguments import get_args
from env import make_vec_envs
from utils.storage import GlobalRolloutStorage, FIFOMemory
from utils.optimization import get_optimizer
from model import RL_Policy, Local_IL_Policy, Neural_SLAM_Module, DdppoPolicy
from utils.utils_for_main import get_local_map_boundaries, init_map_and_pose, \
                                 get_map_from_envs, run_local_planner, visualize_map
import algo

import sys
import matplotlib

if sys.platform == 'darwin':
    matplotlib.use("tkagg")
import matplotlib.pyplot as plt

from tqdm import trange
from torch.utils.tensorboard import SummaryWriter

# for neural implicit mapping 
from env.habitat.exploration_env import load_config 

# heuristic imports
from utils.heuristics import HeuristicTracker

args = get_args()

np.random.seed(args.seed)
torch.manual_seed(args.seed)

if args.cuda:
    torch.cuda.manual_seed(args.seed)


def main():
    # Setup Logging
    log_dir = "{}/models/{}/".format(args.dump_location, args.exp_name)
    dump_dir = "{}/dump/{}/".format(args.dump_location, args.exp_name)
    
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    if not os.path.exists("{}/images/".format(dump_dir)):
        os.makedirs("{}/images/".format(dump_dir))

    logging.basicConfig(
        filename=log_dir + 'train.log',
        level=logging.INFO)
    print("Dumping at {}".format(log_dir))
    logging.info(args)

    # Setup tensorboard 
    writer = SummaryWriter("{}/tensorboard/{}/".format(args.dump_location, args.exp_name))

    # Logging and loss variables
    num_scenes = args.num_processes
    num_episodes = int(args.num_episodes)
    device = args.device  
    policy_loss = 0

    best_cost = 100000
    costs = deque(maxlen=1000)
    exp_costs = deque(maxlen=1000)
    pose_costs = deque(maxlen=1000)

    g_masks = torch.ones(num_scenes).float().to(device)
    l_masks = torch.zeros(num_scenes).float().to(device)

    best_local_loss = np.inf
    best_g_reward = -np.inf

    g_episode_rewards = deque(maxlen=1000)

    l_action_losses = deque(maxlen=1000)

    g_value_losses = deque(maxlen=1000)
    g_action_losses = deque(maxlen=1000)
    g_dist_entropies = deque(maxlen=1000)

    per_step_g_rewards = deque(maxlen=1000)

    g_process_rewards = np.zeros((num_scenes))
    accumulated_ratio = np.zeros((num_scenes))
    all_accumulated_ratios = {}
    uncertainty_list = {}
 
    episode_area_coverage = deque(maxlen=1000)
    per_step_area_coverage = deque(maxlen=1000)

    # Starting environments
    # torch.set_num_threads(1)
    """ 
        create envs
        all the 'computing map for test', semantic warning, 
        optional features, and plugin manager warnings come from here 
    """
    envs = make_vec_envs(args)
    """ 
        call reset() method of Exploration_Env() class for each process
        where: env/habitat/exploration_env.py
        what: randomize episode, get gt map, initialize explored area, curr_loc etc
        stuffs: will call _get_gt_map(). give message 'Computing map for '
    """
    obs, infos = envs.reset()

    if args.debug:
        plt.imshow(np.uint8(np.transpose(obs.cpu().numpy()[0,:,:,:], (1,2,0))))
        plt.savefig('./debug/obs_in_main.png')


    # Initialize map variables
    ### Full map consists of 4 channels containing the following:
    ### 1. Obstacle Map
    ### 2. Exploread Area
    ### 3. Current Agent Location
    ### 4. Past Agent Locations

    torch.set_grad_enabled(False)

    # Calculating full and local map sizes
    map_size = args.map_size_cm // args.map_resolution
    full_w, full_h = map_size, map_size
    local_w, local_h = int(full_w / args.global_downscaling), \
                       int(full_h / args.global_downscaling)

    # Initializing full and local map
    full_map = torch.zeros(num_scenes, 4, full_w, full_h).float().to(device)
    local_map = torch.zeros(num_scenes, 4, local_w, local_h).float().to(device)

    # Initial full and local pose. pose (x, y, o)
    # x,y: xy coordinate in meters 
    # o: orientation (clockwise from x) in deg
    full_pose = torch.zeros(num_scenes, 3).float().to(device)
    local_pose = torch.zeros(num_scenes, 3).float().to(device)

    # Origin of local map
    origins = np.zeros((num_scenes, 3))

    # Local Map Boundaries
    lmb = np.zeros((num_scenes, 4)).astype(int)

    ### Planner pose inputs has 7 dimensions
    ### 1-3 store continuous global agent location
    ### 4-7 store local map boundaries
    planner_pose_inputs = np.zeros((num_scenes, 7))

    full_map, full_pose, local_map, local_pose, planner_pose_inputs, lmb \
        = init_map_and_pose(    full_map, full_pose, local_map, local_pose, planner_pose_inputs, origins, lmb,
                                local_w, local_h, full_w, full_h, 
                                args.map_size_cm, args.map_resolution, args.global_downscaling,
                                num_scenes, device)

    g_observation_space = gym.spaces.Box(0, 1,
                                            (8,
                                            local_w,
                                            local_h), dtype='uint8')

    # Global policy action space
    # action space is (x,y) coordinate
    # with the same bound [0,1] for both dimensions
    g_action_space = gym.spaces.Box(low=0.0, high=1.0,
                                    shape=(2,), dtype=np.float32)

    # Local policy observation space
    l_observation_space = gym.spaces.Box(0, 255,
                                         (3,
                                          args.frame_width,
                                          args.frame_width), dtype='uint8')

    # Local and Global policy recurrent layer sizes
    l_hidden_size = args.local_hidden_size
    g_hidden_size = args.global_hidden_size

    # slam
    nslam_module = Neural_SLAM_Module(args).to(device)
    slam_optimizer = get_optimizer(nslam_module.parameters(),
                                   args.slam_optimizer)

    # Global policy
    if args.global_arch == 'NeuralSLAM':
        g_policy = RL_Policy(g_observation_space.shape, g_action_space,
                            model_type='NeuralSLAM',
                            base_kwargs={'recurrent': args.use_recurrent_global,
                                        'hidden_size': g_hidden_size,
                                        'downscaling': args.global_downscaling
                                        }).to(device)
    elif args.global_arch == 'lena':
        g_policy = RL_Policy(g_observation_space.shape, g_action_space,
                            model_type='lena',
                            base_kwargs={'recurrent': args.use_recurrent_global,
                                        'hidden_size': g_hidden_size,
                                        'downscaling': args.global_downscaling
                                        }).to(device)
    else:
        raise NotImplementedError
    g_agent = algo.PPO(g_policy, args.clip_param, args.ppo_epoch,
                       args.num_mini_batch, args.value_loss_coef,
                       args.entropy_coef, lr=args.global_lr, eps=args.eps,
                       max_grad_norm=args.max_grad_norm)

    # Local policy
    if args.use_DD_PPO == 'none':
        l_policy = Local_IL_Policy(l_observation_space.shape, 3,
                                recurrent=args.use_recurrent_local,
                                hidden_size=l_hidden_size,
                                deterministic=args.use_deterministic_local).to(device)
        local_optimizer = get_optimizer(l_policy.parameters(),
                                        args.local_optimizer)
    else:
        print(f'load DD-PPO policy from {args.use_DD_PPO}')
        l_policy = DdppoPolicy(path=args.use_DD_PPO, 
                               num_scenes=num_scenes).to(device)

    # Storage
    g_rollouts = GlobalRolloutStorage(args.num_global_steps,
                                      num_scenes, g_observation_space.shape,
                                      g_action_space, g_policy.rec_state_size,
                                      1).to(device)

    slam_memory = FIFOMemory(args.slam_memory_size)

    # Loading model
    if args.load_slam != "0":
        print("Loading slam {}".format(args.load_slam))
        state_dict = torch.load(args.load_slam,
                                map_location=lambda storage, loc: storage)
        nslam_module.load_state_dict(state_dict)

    if not args.train_slam:
        nslam_module.eval()

    if args.load_global != "0":
        print("Loading global {}".format(args.load_global))
        state_dict = torch.load(args.load_global,
                                map_location=lambda storage, loc: storage)
        g_policy.load_state_dict(state_dict)

    if not args.train_global:
        g_policy.eval()

    if args.load_local != "0" and args.use_DD_PPO == 'none':
        print("Loading local {}".format(args.load_local))
        state_dict = torch.load(args.load_local,
                                map_location=lambda storage, loc: storage)
        l_policy.load_state_dict(state_dict)

    if not args.train_local and args.use_DD_PPO == 'none':
        l_policy.eval()

    # Predict map from frame 1:
    # output obstacle map, explored area, and pose
    
    if args.use_nslam:
        poses = torch.from_numpy(np.asarray(
            [infos[env_idx]['sensor_pose'] for env_idx
            in range(num_scenes)])).float().to(device)
        _, _, local_map[:, 0, :, :], local_map[:, 1, :, :], _, local_pose = \
            nslam_module(obs, obs, poses, local_map[:, 0, :, :],
                        local_map[:, 1, :, :], local_pose)
        if args.debug:
            gt_map = infos[0]['map'][lmb[0, 0]:lmb[0, 1], lmb[0, 2]:lmb[0, 3]]
            plt.figure()
            plt.subplot(1,3,1)
            plt.imshow(local_map[0, 0, :, :].cpu().numpy())
            plt.subplot(1,3,2)
            plt.imshow(gt_map)
            plt.subplot(1,3,3)
            plt.imshow(gt_map*8.0 + 5.0*local_map[0, 0, :, :].cpu().numpy())
            plt.savefig('./debug/nslam.png')
            print(f'nslam pose = {local_pose}')
            print(f"gt pose = { infos[0]['sensor_pose'] }")
    else:
        local_map, local_pose = get_map_from_envs(infos, local_map, local_pose, lmb, origins,
                                                  num_scenes, args.global_arch, device)

    # Compute Global policy input
    locs = local_pose.cpu().numpy()
    global_orientation = torch.zeros(num_scenes, 1).long()

    global_input = torch.zeros(num_scenes, 8, local_w, local_h)
    for e in range(num_scenes):
        r, c = locs[e, 1], locs[e, 0] # position row and column 
        loc_r, loc_c = [int(r * 100.0 / args.map_resolution),
                        int(c * 100.0 / args.map_resolution)] # convert m to cm and find current pose bin
        local_map[e, 2:, loc_r - 1:loc_r + 2, loc_c - 1:loc_c + 2] = 1. # occupy the position at the current and past agent position channels
        global_orientation[e] = int((locs[e, 2] + 180.0) / 5.) # -180~180 to 0~360, digitize

    global_input[:, 0:4, :, :] = local_map.detach()
    global_input[:, 4:, :, :] = nn.MaxPool2d(args.global_downscaling)(full_map)

    g_rollouts.obs[0].copy_(global_input)
    g_rollouts.extras[0].copy_(global_orientation)

    # Run Global Policy (global_goals = Long-Term Goal)
    g_value, g_action, g_action_log_prob, g_rec_states = \
        g_policy.act(
            g_rollouts.obs[0],
            g_rollouts.rec_states[0],
            g_rollouts.masks[0],
            extras=g_rollouts.extras[0],
            deterministic=False
        )

    cpu_actions = nn.Sigmoid()(g_action).cpu().numpy()
    global_goals = [[int(action[0] * local_w/2), int(action[1] * local_h/2)]
                    for action in cpu_actions] # a list of xy coordinates # TODO
        

    last_obs = obs.detach()
    local_rec_states = torch.zeros(num_scenes, l_hidden_size).to(device)

    total_num_steps = -1
    g_reward = 0

    torch.set_grad_enabled(False)

    # Initialize variables to check for stuck agents
    heuristic_tracker = HeuristicTracker(args, num_scenes)

    for ep_num in trange(num_episodes):
        all_accumulated_ratios[ep_num] = []
        uncertainty_list[ep_num] = []

        if args.use_DD_PPO != 'none':
            l_policy.reset()
        for step in trange(args.max_episode_length):
            total_num_steps += 1

            g_step = (step // args.num_local_steps) % args.num_global_steps
            l_step = step % args.num_local_steps

            # ------------------------------------------------------------------
            # Check for stuck agents
            heuristic_tracker.update(local_pose)

            # ------------------------------------------------------------------
            # Local Planner and Policy
            # use ground truth local planner
            if args.use_ffm_planner == 0:
                if args.train_local:
                    torch.set_grad_enabled(True)
            
                # train local planner 
                if args.use_DD_PPO == 'none':
                    output = run_local_planner(num_scenes, 
                            global_goals, global_input, planner_pose_inputs,
                            envs)

                    del last_obs
                    last_obs = obs.detach()
                    local_masks = l_masks
                    local_goals = output[:, :-1].to(device).long()

                    if args.train_local:
                        torch.set_grad_enabled(True)

                    action, action_prob, local_rec_states = l_policy(
                        obs,
                        local_rec_states,
                        local_masks,
                        extras=local_goals,
                    )
                    if args.train_local:
                        action_target = output[:, -1].long().to(device)
                        policy_loss += nn.CrossEntropyLoss()(action_prob, action_target)
                        torch.set_grad_enabled(False)
                    l_action = action.cpu()
                    l_action += 1

                # train with DDPPO local policy
                else:
                    assert type(l_policy) == DdppoPolicy
                    l_action = l_policy.plan(np.stack([infos[e]['depth'] for e in range(num_scenes)]) , 
                                            global_goals, planner_pose_inputs, 
                                            l_masks, step, 
                                            args.map_resolution, device)
                    
            # use ground truth local planner from NSLAM codebase
            else:
                with torch.no_grad():
                    output = run_local_planner(num_scenes, 
                                global_goals, global_input, planner_pose_inputs,
                                envs)
                    l_action = output[:,2].cpu().long()    
            
            # for visualization
            if args.print_images:
                visualize_map(num_scenes, 
                              global_goals, global_input, planner_pose_inputs,
                              envs, heuristic=heuristic_tracker._get_stuck())

            # --------------------------------------------------------------
            # Apply heuristics

            print("Local Action: ", l_action, l_action.shape)
            heuristic_actions = heuristic_tracker.get_heuristic_actions(local_pose)
            for i, a in enumerate(heuristic_actions):
                if a and a != -1:
                    print("Heuristic Action, ", a, " applied for agent ", i)
                    l_action[i][0] = a
            print("Modified Action: ", l_action, l_action.shape)

            # ------------------------------------------------------------------
            # Env step
            #l_action *= 0 # TODO: DON't MOVE!
            # if step < 1000:
            #     l_action = l_action*0 + 3 # TODO: spinnnnnnn right
            obs, rew, done, infos = envs.step(l_action)

            l_masks = torch.FloatTensor([0 if x else 1
                                         for x in done]).to(device)
            g_masks *= l_masks # mask = 0 for done scenes 
            # ------------------------------------------------------------------

            # ------------------------------------------------------------------
            # Reinitialize variables when episode ends
            if step == args.max_episode_length - 1:  # Last episode step
                full_map, full_pose, local_map, local_pose, planner_pose_inputs, lmb \
                    = init_map_and_pose(    full_map, full_pose, local_map, local_pose, planner_pose_inputs, origins, lmb,
                                            local_w, local_h, full_w, full_h, 
                                            args.map_size_cm, args.map_resolution, args.global_downscaling,
                                            num_scenes, device)
                del last_obs
                last_obs = obs.detach()
            # ------------------------------------------------------------------

            # ------------------------------------------------------------------
            # Neural SLAM Module
            if args.use_nslam:
                poses = torch.from_numpy(np.asarray(
                    [infos[env_idx]['sensor_pose'] for env_idx
                    in range(num_scenes)]) ).float().to(device)
                if args.train_slam:
                    # Add frames to memory
                    for env_idx in range(num_scenes):
                        env_obs = obs[env_idx].to("cpu")
                        env_poses = torch.from_numpy(np.asarray(
                            infos[env_idx]['sensor_pose']
                        )).float().to("cpu")
                        env_gt_fp_projs = torch.from_numpy(np.asarray(
                            infos[env_idx]['fp_proj']
                        )).unsqueeze(0).float().to("cpu")
                        env_gt_fp_explored = torch.from_numpy(np.asarray(
                            infos[env_idx]['fp_explored']
                        )).unsqueeze(0).float().to("cpu")
                        env_gt_pose_err = torch.from_numpy(np.asarray(
                            infos[env_idx]['pose_err']
                        )).float().to("cpu")
                        slam_memory.push(
                            (last_obs[env_idx].cpu(), env_obs, env_poses),
                            (env_gt_fp_projs, env_gt_fp_explored, env_gt_pose_err))

                _, _, local_map[:, 0, :, :], local_map[:, 1, :, :], _, local_pose = \
                    nslam_module(last_obs, obs, poses, local_map[:, 0, :, :],
                                local_map[:, 1, :, :], local_pose, build_maps=True)
                
            else:
                local_map, local_pose = get_map_from_envs(infos, local_map, local_pose, lmb, origins,
                                                  num_scenes, args.global_arch, device)
            
            locs = local_pose.cpu().numpy()
            planner_pose_inputs[:, :3] = locs + origins # Neural SLAM estimates locs w.r.t local window 
            local_map[:, 2, :, :].fill_(0.)  # Resetting current location channel

            for e in range(num_scenes):
                r, c = locs[e, 1], locs[e, 0]
                loc_r, loc_c = [int(r * 100.0 / args.map_resolution),
                                int(c * 100.0 / args.map_resolution)]
                local_map[e, 2:, loc_r - 2:loc_r + 3, loc_c - 2:loc_c + 3] = 1.
            # ------------------------------------------------------------------

            # ------------------------------------------------------------------
            # Global Policy
            if l_step == args.num_local_steps - 1:
                # For every global step, update the full and local maps
                for e in range(num_scenes):
                    full_map[e, :, lmb[e, 2]:lmb[e, 3], lmb[e, 0]:lmb[e, 1]] = \
                        local_map[e] # update full map
                    full_pose[e] = local_pose[e] + \
                                   torch.from_numpy(origins[e]).to(device).float()

                    locs = full_pose[e].cpu().numpy()
                    r, c = locs[1], locs[0]
                    loc_r, loc_c = [int(r * 100.0 / args.map_resolution),
                                    int(c * 100.0 / args.map_resolution)]

                    lmb[e] = get_local_map_boundaries((loc_r, loc_c),
                                                      (local_w, local_h),
                                                      (full_w, full_h),
                                                      args.global_downscaling) # get new lmb

                    planner_pose_inputs[e, 3:] = lmb[e]
                    origins[e] = [lmb[e][0] * args.map_resolution / 100.0,
                                  lmb[e][2] * args.map_resolution / 100.0, 0.]

                    local_map[e] = full_map[e, :,
                                   lmb[e, 2]:lmb[e, 3], lmb[e, 0]:lmb[e, 1]] # local map with new boundaires
                    local_pose[e] = full_pose[e] - \
                                    torch.from_numpy(origins[e]).to(device).float()

                locs = local_pose.cpu().numpy()
                for e in range(num_scenes):
                    global_orientation[e] = np.clip(int((locs[e, 2] + 180.0) / 5.), 0, 71)

                global_input[:, 0:4, :, :] = local_map
                global_input[:, 4:, :, :] = \
                    nn.MaxPool2d(args.global_downscaling)(full_map)

                # Get exploration reward and metrics
                g_reward = torch.from_numpy(np.asarray(
                    [infos[env_idx]['exp_reward'] for env_idx
                     in range(num_scenes)])
                ).float().to(device)

                g_process_rewards += g_reward.cpu().numpy()
                g_total_rewards = g_process_rewards * \
                                  (1 - g_masks.cpu().numpy()) # only for done scenes 
                g_process_rewards *= g_masks.cpu().numpy() # set accumlated rewards to zero for the scenes that are done 
                per_step_g_rewards.append(np.mean(g_reward.cpu().numpy()))

                if np.sum(g_total_rewards) != 0:
                    for tr in g_total_rewards:
                        g_episode_rewards.append(tr) if tr != 0 else None
               
                exp_ratio = np.asarray([infos[env_idx]['exp_ratio'] for env_idx in range(num_scenes)]) 
                per_step_area_coverage.append(np.mean(exp_ratio))
                accumulated_ratio += exp_ratio
                all_accumulated_ratios[ep_num].append(accumulated_ratio.tolist())
                done_ratio = accumulated_ratio * (1 - g_masks.cpu().numpy()) # only for done scenes 
                accumulated_ratio *= g_masks.cpu().numpy() # set done scenes exp ratio to zero

                if args.use_NeRF_mapping:
                    uncertainty = np.asarray([infos[env_idx]['uncertainty'] for env_idx in range(num_scenes)]) 
                    uncertainty_list[ep_num].append(uncertainty.tolist())

                if np.sum(done_ratio) != 0:
                    for scene_ratio in done_ratio:
                        episode_area_coverage.append(scene_ratio) if scene_ratio != 0 else None

                # Add samples to global policy storage
                g_rollouts.insert(
                    global_input, g_rec_states,
                    g_action, g_action_log_prob, g_value,
                    g_reward, g_masks, global_orientation
                )

                # Sample long-term goal from global policy
                g_value, g_action, g_action_log_prob, g_rec_states = \
                    g_policy.act(
                        g_rollouts.obs[g_step + 1],
                        g_rollouts.rec_states[g_step + 1],
                        g_rollouts.masks[g_step + 1],
                        extras=g_rollouts.extras[g_step + 1],
                        deterministic=False
                    )
                cpu_actions = nn.Sigmoid()(g_action).cpu().numpy()
                global_goals = [[int(action[0] * local_w),
                                 int(action[1] * local_h)]
                                for action in cpu_actions] 

                g_reward = 0
                g_masks = torch.ones(num_scenes).float().to(device)
            # ------------------------------------------------------------------

            ### TRAINING
            torch.set_grad_enabled(True)
            # ------------------------------------------------------------------
            # Train Neural SLAM Module
            if args.train_slam and len(slam_memory) > args.slam_batch_size:
                for _ in range(args.slam_iterations):
                    inputs, outputs = slam_memory.sample(args.slam_batch_size)
                    b_obs_last, b_obs, b_poses = inputs
                    gt_fp_projs, gt_fp_explored, gt_pose_err = outputs

                    b_obs = b_obs.to(device)
                    b_obs_last = b_obs_last.to(device)
                    b_poses = b_poses.to(device)

                    gt_fp_projs = gt_fp_projs.to(device)
                    gt_fp_explored = gt_fp_explored.to(device)
                    gt_pose_err = gt_pose_err.to(device)

                    b_proj_pred, b_fp_exp_pred, _, _, b_pose_err_pred, _ = \
                        nslam_module(b_obs_last, b_obs, b_poses,
                                     None, None, None,
                                     build_maps=False)
                    loss = 0
                    if args.proj_loss_coeff > 0:
                        proj_loss = F.binary_cross_entropy(b_proj_pred,
                                                           gt_fp_projs)
                        costs.append(proj_loss.item())
                        loss += args.proj_loss_coeff * proj_loss

                    if args.exp_loss_coeff > 0:
                        exp_loss = F.binary_cross_entropy(b_fp_exp_pred,
                                                          gt_fp_explored)
                        exp_costs.append(exp_loss.item())
                        loss += args.exp_loss_coeff * exp_loss

                    if args.pose_loss_coeff > 0:
                        pose_loss = torch.nn.MSELoss()(b_pose_err_pred,
                                                       gt_pose_err)
                        pose_costs.append(args.pose_loss_coeff *
                                          pose_loss.item())
                        loss += args.pose_loss_coeff * pose_loss

                    if args.train_slam:
                        slam_optimizer.zero_grad()
                        loss.backward()
                        slam_optimizer.step()

                    del b_obs_last, b_obs, b_poses
                    del gt_fp_projs, gt_fp_explored, gt_pose_err
                    del b_proj_pred, b_fp_exp_pred, b_pose_err_pred

            # ------------------------------------------------------------------

            # ------------------------------------------------------------------
            # Train Local Policy
            if (l_step + 1) % args.local_policy_update_freq == 0 \
                    and args.train_local:
                local_optimizer.zero_grad()
                policy_loss.backward()
                local_optimizer.step()
                l_action_losses.append(policy_loss.item())
                policy_loss = 0
                local_rec_states = local_rec_states.detach_()
            # ------------------------------------------------------------------

            # ------------------------------------------------------------------
            # Train Global Policy
            if g_step == args.num_global_steps - 1 \
                    and l_step == args.num_local_steps - 1:
                if args.train_global:
                    g_next_value = g_policy.get_value(
                        g_rollouts.obs[-1],
                        g_rollouts.rec_states[-1],
                        g_rollouts.masks[-1],
                        extras=g_rollouts.extras[-1]
                    ).detach()

                    g_rollouts.compute_returns(g_next_value, args.use_gae,
                                               args.gamma, args.tau)
                    g_value_loss, g_action_loss, g_dist_entropy = \
                        g_agent.update(g_rollouts)
                    g_value_losses.append(g_value_loss)
                    g_action_losses.append(g_action_loss)
                    g_dist_entropies.append(g_dist_entropy)
                g_rollouts.after_update()
            # ------------------------------------------------------------------

            # Finish Training
            torch.set_grad_enabled(False)
            # ------------------------------------------------------------------

            # ------------------------------------------------------------------
            # Logging
            if total_num_steps % args.log_interval == 0:
                writer.add_scalar('global reward/step mean', np.mean(per_step_g_rewards), total_num_steps)
                writer.add_scalar('global reward/ep mean', np.mean(g_episode_rewards), total_num_steps)

                writer.add_scalar('global area coverage/ep mean', 
                                  np.mean(episode_area_coverage), total_num_steps)
                writer.add_scalar('global area coverage/step mean', 
                                  np.mean(per_step_area_coverage), total_num_steps)

                writer.add_scalar('global loss/value', np.mean(g_value_losses), total_num_steps)
                writer.add_scalar('global loss/action', np.mean(g_action_losses), total_num_steps)
                writer.add_scalar('global loss/dist', np.mean(g_dist_entropies), total_num_steps)

                #torch.cuda.memory._dump_snapshot( os.path.join(log_dir, "GPU_memory.pickle") )

            # ------------------------------------------------------------------

            # ------------------------------------------------------------------
            # Save best models
            if (total_num_steps * num_scenes) % args.save_interval < \
                    num_scenes:

                # Save Neural SLAM Model
                if len(costs) >= 1000 and np.mean(costs) < best_cost \
                        and not args.eval:
                    best_cost = np.mean(costs)
                    torch.save(nslam_module.state_dict(),
                               os.path.join(log_dir, "model_best.slam"))

                # Save Local Policy Model
                if len(l_action_losses) >= 100 and \
                        (np.mean(l_action_losses) <= best_local_loss) \
                        and not args.eval:
                    torch.save(l_policy.state_dict(),
                               os.path.join(log_dir, "model_best.local"))

                    best_local_loss = np.mean(l_action_losses)

                # Save Global Policy Model
                if len(g_episode_rewards) >= 100 and \
                        (np.mean(g_episode_rewards) >= best_g_reward) \
                        and not args.eval:
                    torch.save(g_policy.state_dict(),
                               os.path.join(log_dir, "model_best.global"))
                    best_g_reward = np.mean(g_episode_rewards)

            # Save periodic models
            if (total_num_steps * num_scenes) % args.save_periodic < \
                    num_scenes:
                step = total_num_steps * num_scenes
                if args.train_slam:
                    torch.save(nslam_module.state_dict(),
                               os.path.join(dump_dir,
                                            "periodic_{}.slam".format(step)))
                if args.train_local:
                    torch.save(l_policy.state_dict(),
                               os.path.join(dump_dir,
                                            "periodic_{}.local".format(step)))
                if args.train_global:
                    torch.save(g_policy.state_dict(),
                               os.path.join(dump_dir,
                                            "periodic_{}.global".format(step)))
            # ------------------------------------------------------------------
        
        # generate video out of images when an episode ends
        if args.print_images:
            for scene in range(num_scenes):
                img_path = '{}/thread_{}/ep_{}/%04d.png'.format(dump_dir, scene+1, (ep_num+1)*2)
                save_path = '{}/thread_{}/video_{}.mp4'.format(dump_dir, scene+1, (ep_num+1)*2)
                os.system(
                    f"ffmpeg -framerate 30  -i  {img_path} -y {save_path}")

    if args.eval:
        with open(os.path.join(dump_dir, 'accumulated_ratios.json'), 'w') as f:
            json.dump(all_accumulated_ratios, f, indent=4)
        if args.use_NeRF_mapping:
            with open(os.path.join(dump_dir, 'uncertainty_history.json'), 'w') as f:
                json.dump(uncertainty_list, f, indent=4)


if __name__ == "__main__":
    print("Starting Neural SLAM Training")
    main()
