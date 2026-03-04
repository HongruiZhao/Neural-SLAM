import os
import numpy as np
import torch
import torch.nn as nn
from arguments import get_args
from env import make_vec_envs
from utils.global_policy_handler import GlobalPolicyHandler
from utils.mapping_handler import MappingHandler
from utils.local_policy_handler import LocalPolicyHandler
from utils.utils_for_main import visualize_map, TensorboardLogger

from tqdm import trange
from torch.utils.tensorboard import SummaryWriter

# heuristic imports
from utils.heuristics import HeuristicTracker

args = get_args()
torch.set_float32_matmul_precision('medium') 
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
    if not os.path.exists(dump_dir):
        os.makedirs(dump_dir)

    writer = SummaryWriter("{}/tensorboard/{}/".format(args.dump_location, args.exp_name))
    logger = TensorboardLogger(writer)

    num_scenes = args.num_processes
    num_episodes = int(args.num_episodes)
    device = args.device  

    envs = make_vec_envs(args)
    
    m_handler = MappingHandler(args, num_scenes, device)
    g_handler = GlobalPolicyHandler(args, num_scenes, device, logger)
    l_handler = LocalPolicyHandler(args, num_scenes, device)
    
    torch.set_grad_enabled(False) 

    # Loading models
    g_handler.load_model(args.load_global)
    if not args.train_global:
        g_handler.eval()

    total_num_steps = 0

    # Initialize variables to check for stuck agents
    heuristic_tracker = HeuristicTracker(args, num_scenes)


    for ep_num in trange(num_episodes, smoothing=0):
        # Manual reset at the start of each episode
        _, infos = envs.reset()

        # Initializing maps and pose for the new episode
        m_handler._init_map_and_pose()
        
        # Predict map from frame 1 and compute global input
        m_handler.get_full_map_pose(infos, is_global_step=True)
        m_handler.get_local_map_pose()
        global_input, global_orientation = m_handler.get_global_input()

        # Initialize rollout storage and reset handlers
        g_handler.init_rollout(global_input, global_orientation)
        g_handler.reset_episode(ep_num)
        # Run Global Policy (global_goals = Long-Term Goal)
        global_goals = g_handler.act(0)
        global_goals_m = g_handler.get_global_goals_m(global_goals, m_handler.planner_pose_inputs)
        envs.set_global_goals(global_goals_m)
        
        g_handler.log_value(0)
        l_handler.reset()


        for step in trange(args.max_episode_length, smoothing=0):
            g_step = (step // args.num_local_steps) % args.num_global_steps
            l_step = step % args.num_local_steps

            if args.heuristic_strategy != 'none':
                heuristic_tracker.update(m_handler.local_pose)

            l_action = l_handler.plan(infos, global_goals, m_handler.planner_pose_inputs, 
                                     g_handler.g_masks, step)
            
            if args.heuristic_strategy != 'none':
                heuristic_actions = heuristic_tracker.get_heuristic_actions(m_handler.local_pose)
                for i, a in enumerate(heuristic_actions):
                    if a and a != -1:
                        l_action[i][0] = a
            
            if args.print_images:
                visualize_map(num_scenes, 
                              global_goals, global_input, m_handler.planner_pose_inputs,
                              envs, full_map=m_handler.full_map, local_map=m_handler.local_map,
                              heuristic=heuristic_tracker._get_stuck(),
                              value_history=g_handler.g_value_history)

            _, _, done, infos = envs.step(l_action)
            g_handler.update_masks(done)

            # Update mapping
            is_global_step = (l_step == args.num_local_steps - 1)
            m_handler.get_full_map_pose(infos, is_global_step=is_global_step)
            m_handler.get_local_map_pose()

            if l_step == args.num_local_steps - 1:
                global_input, global_orientation = m_handler.get_global_input()
                g_reward = g_handler.process_rewards(infos, ep_num)
                g_handler.insert_rollout(
                    global_input, g_reward, global_orientation
                )
                global_goals = g_handler.act(g_step + 1)
                global_goals_m = g_handler.get_global_goals_m(global_goals, m_handler.planner_pose_inputs)
                envs.set_global_goals(global_goals_m)
                
                g_handler.log_value(step + 1)

            ### TRAINING
            torch.set_grad_enabled(True)
            if g_step == args.num_global_steps - 1 and l_step == args.num_local_steps - 1:
                if args.train_global:
                    g_handler.train()
            torch.set_grad_enabled(False)

            # logging and save
            if total_num_steps % args.log_interval == 0:
                logger.write(total_num_steps)

            if (total_num_steps * num_scenes) % args.save_periodic < num_scenes:
                step_idx = total_num_steps * num_scenes
                if args.train_global:
                    g_handler.save_model(step_idx, dump_dir)

            total_num_steps += 1
        
        # generate video out of images when an episode ends
        if args.print_images:
            for scene in range(num_scenes):
                ep_dir_num = infos[scene]['episode_no']
                img_path = '{}/thread_{}/ep_{}/%04d.png'.format(dump_dir, scene+1, ep_dir_num)
                save_path = '{}/thread_{}/video_{}.mp4'.format(dump_dir, scene+1, ep_dir_num)
                os.system(
                    f"ffmpeg -framerate 30  -i  {img_path} -y {save_path}")


    if args.eval:
        g_handler.save_results(dump_dir)


if __name__ == "__main__":
    print("Starting Neural SLAM Training")
    main()
