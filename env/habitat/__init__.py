# Parts of the code in this file have been borrowed from:
#    https://github.com/facebookresearch/habitat-api

import numpy as np
import torch

from .exploration_env import Exploration_Env


import habitat
from habitat.config.default import get_config as cfg_env
from habitat.datasets.pointnav.pointnav_dataset import PointNavDatasetV1
from habitat.core.vector_env import VectorEnv # creates multiple processes where each process runs its own environment
from habitat_baselines.config.default import get_config as cfg_baseline

import omegaconf

def make_env_fn(args, config_env, config_baseline, rank):
    dataset = PointNavDatasetV1(config_env.habitat.dataset)
    omegaconf.OmegaConf.set_readonly(config_env, False)
    config_env.habitat.simulator.scene = dataset.episodes[0].scene_id
    print("Loading {}".format(config_env.habitat.simulator.scene))
    omegaconf.OmegaConf.set_readonly(config_env, True)

    env = Exploration_Env(args=args, rank=rank,
                          config_env=config_env, config_baseline=config_baseline, dataset=dataset
                          )
    env.seed(rank)

    return env

def construct_envs(args):
    env_configs = []
    baseline_configs = []
    args_list = []

    basic_config = cfg_env(config_path=
                           "env/habitat/configs/" + args.task_config)
    
    omegaconf.OmegaConf.set_readonly(basic_config, False)
    basic_config.habitat.dataset.split = args.split
    omegaconf.OmegaConf.set_readonly(basic_config, True)


    scenes = PointNavDatasetV1.get_scenes_to_load(basic_config.habitat.dataset)
    print(f'num of scenes = {len(scenes)}, scenes = {scenes}')
    if len(scenes) > 0:
        assert len(scenes) >= args.num_processes, (
            "reduce the number of processes as there "
            "aren't enough number of scenes"
        )
        scene_split_size = int(np.floor(len(scenes) / args.num_processes))


    for i in range(args.num_processes):
        config_env = cfg_env(config_path=
                           "env/habitat/configs/" + args.task_config)
        omegaconf.OmegaConf.set_readonly(config_env, False)

        if len(scenes) > 0:
            config_env.habitat.dataset.content_scenes = scenes[
                                                i * scene_split_size: (i + 1) * scene_split_size
                                                ]

        if i < args.num_processes_on_first_gpu:
            gpu_id = 0
        else:
            gpu_id = int((i - args.num_processes_on_first_gpu)
                         // args.num_processes_per_gpu) + args.sim_gpu_id
        gpu_id = min(torch.cuda.device_count() - 1, gpu_id)
        config_env.habitat.simulator.habitat_sim_v0.gpu_device_id = gpu_id 

        config_env.habitat.environment.max_episode_steps = args.max_episode_length
        config_env.habitat.environment.iterator_options.shuffle = False

        config_env.habitat.simulator.agents.main_agent.sim_sensors.rgb_sensor.width = args.env_frame_width
        config_env.habitat.simulator.agents.main_agent.sim_sensors.rgb_sensor.height = args.env_frame_height
        config_env.habitat.simulator.agents.main_agent.sim_sensors.rgb_sensor.hfov = int(args.hfov)
        config_env.habitat.simulator.agents.main_agent.sim_sensors.rgb_sensor.position = [0, args.camera_height, 0]
        config_env.habitat.simulator.agents.main_agent.sim_sensors.depth_sensor.position = [0, args.camera_height, 0]

        config_env.habitat.simulator.turn_angle = 10
        config_env.habitat.dataset.split = args.split

        omegaconf.OmegaConf.set_readonly(config_env, True)
        env_configs.append(config_env)

        config_baseline = cfg_baseline(config_path=
                                       "env/habitat/configs/ppo_pointnav.yaml" )
        baseline_configs.append(config_baseline)

        args_list.append(args)

    if args.debug:
        envs = habitat.ThreadedVectorEnv(
            make_env_fn=make_env_fn,
            env_fn_args=tuple(zip(args_list, env_configs, baseline_configs, 
                                  range(args.num_processes))),
    )
    else:
        envs = VectorEnv(
            make_env_fn=make_env_fn,
            env_fn_args=tuple(zip(args_list, env_configs, baseline_configs, 
                                  range(args.num_processes))),
        )


    return envs
