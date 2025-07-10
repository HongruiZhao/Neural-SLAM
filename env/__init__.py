import torch
import numpy as np
from .habitat import construct_envs


def make_vec_envs(args):
    envs = construct_envs(args)
    envs = VecPyTorch(envs, args.device)
    return envs


# Adapted from https://github.com/ikostrikov/pytorch-a2c-ppo-acktr-gail/blob/master/a2c_ppo_acktr/envs.py#L159
class VecPyTorch():

    def __init__(self, venv, device):
        self.venv = venv
        self.num_envs = venv.num_envs
        self.observation_space = venv.observation_spaces
        self.action_space = venv.action_spaces
        self.device = device

    def reset(self):
        output_list = self.venv.reset()

        obs_array = np.stack([ item[0]['obs'].copy() for item in output_list ])
        obs_array = torch.from_numpy(obs_array).float().to(self.device)
        info_list = [ item[1] for item in output_list]

        return obs_array, info_list


    def step_async(self, actions):
        actions = actions.cpu().numpy()
        self.venv.async_step(actions)


    def step(self, actions):
        actions = actions.cpu().numpy()
        output_list = self.venv.step(actions)

        try:
            obs_array = np.stack([ item[0]['obs'].copy() for item in output_list ])
        except TypeError: # output somehow changes when 'done' True. first element becomes (obs_dict, info_dict)  
            obs_array = np.stack([ item[0][0]['obs'].copy() for item in output_list ])
        obs_array = torch.from_numpy(obs_array).float().to(self.device)
        reward_array = np.array([item[1] for item in output_list])
        reward_array = torch.from_numpy(reward_array)
        done_list = [ item[2] for item in output_list]
        info_list = [ item[3] for item in output_list]

        return obs_array, reward_array, done_list, info_list

    def get_rewards(self, inputs):
        reward = self.venv.get_rewards(inputs)
        reward = torch.from_numpy(reward).float()
        return reward

    def get_short_term_goal(self, inputs):

        function_args_list = [{'inputs': d} for d in inputs]
        stg = self.venv.call(['get_short_term_goal']*self.num_envs, function_args_list)
        stg = np.stack(stg)
        stg = torch.from_numpy(stg).float()
        return stg

    def close(self):
        return self.venv.close()
