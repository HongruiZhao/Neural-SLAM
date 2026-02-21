import os
import numpy as np
import torch
import gym
import json
from model import RL_Policy
from utils.storage import GlobalRolloutStorage
from utils.utils_for_main import TensorboardLogger
import algo

class GlobalPolicyHandler:
    """
    Class handling the RL training and action sampling of the global policy.
    """
    def __init__(self, args, num_scenes: int, device: torch.device, logger: TensorboardLogger):
        """
        Initialize the GlobalPolicyHandler.
        Args:
            args: Command line arguments.
            num_scenes: Number of parallel environments.
            device: Torch device.
            logger: Tensorboard logger.
        """
        self.args = args
        self.num_scenes = num_scenes
        self.device = device
        self.logger = logger
        
        # Determine local dimensions for observation space
        map_size = args.map_size_cm // args.map_resolution
        self.local_w = int(map_size / args.global_downscaling)
        self.local_h = int(map_size / args.global_downscaling)

        self.g_observation_space = gym.spaces.Box(low=0.0, high=1.0, shape=(8, self.local_w, self.local_h), dtype=np.float32)
        self.g_action_space = gym.spaces.Box(low=0.0, high=1.0, shape=(2,), dtype=np.float32)
        
        if args.global_arch == 'NeuralSLAM':
            self.g_policy = RL_Policy(self.g_observation_space.shape, self.g_action_space,
                                model_type='NeuralSLAM',
                                base_kwargs={'recurrent': args.use_recurrent_global,
                                            'hidden_size': args.global_hidden_size,
                                            'downscaling': args.global_downscaling
                                            }).to(device)
        elif args.global_arch == 'lena':
            self.g_policy = RL_Policy(self.g_observation_space.shape, self.g_action_space,
                                model_type='lena',
                                base_kwargs={'recurrent': args.use_recurrent_global,
                                            'hidden_size': args.global_hidden_size,
                                            'downscaling': args.global_downscaling
                                            }).to(device)
        else:
            raise NotImplementedError
        
        if hasattr(torch, 'compile'):
            self.g_policy = torch.compile(self.g_policy)

        self.g_agent = algo.PPO(self.g_policy, args.clip_param, args.ppo_epoch,
                                args.num_mini_batch, args.value_loss_coef,
                                args.entropy_coef, lr=args.global_lr, eps=args.eps,
                                max_grad_norm=args.max_grad_norm)

        self.g_rollouts = GlobalRolloutStorage( args.num_global_steps,
                                                num_scenes, self.g_observation_space.shape,
                                                self.g_action_space, self.g_policy.rec_state_size,
                                                1).to(device)
                
        # Policy state variables
        self.g_value = None
        self.g_action = None
        self.g_action_log_prob = None
        self.g_rec_states = torch.zeros(num_scenes, self.g_policy.rec_state_size).to(device)

        # Tracking and Logging variables
        self.g_masks = torch.ones(num_scenes).float().to(device)
        self.g_process_rewards = np.zeros(num_scenes)
        self.accumulated_ratio = np.zeros(num_scenes)
        self.all_accumulated_ratios = {}
        self.uncertainty_list = {}
        self.g_value_history = [[] for _ in range(num_scenes)]


    def load_model(self, path: str):
        """
        Load global policy model from path.
        Args:
            path: Path to the model state dict.
        """
        if path != "0":
            print("Loading global {}".format(path))
            state_dict = torch.load(path, map_location=lambda storage, loc: storage)
            #self.g_policy.load_state_dict({k.replace('_orig_mod.', ''): v for k, v in state_dict.items()})
            self.g_policy.load_state_dict(state_dict)


    def eval(self):
        """Set global policy to evaluation mode."""
        self.g_policy.eval()


    def reset_episode(self, ep_num: int):
        """
        Reset tracking variables for a new episode.
        Args:
            ep_num: Current episode number.
        """
        self.all_accumulated_ratios[ep_num] = []
        self.uncertainty_list[ep_num] = []
        self.g_process_rewards.fill(0.)
        self.accumulated_ratio.fill(0.)
        self.g_value_history = [[] for _ in range(self.num_scenes)]


    def log_value(self, step: int):
        """
        Log the current global value prediction for visualization.
        Args:
            step: Current step number.
        """
        for e in range(self.num_scenes):
            self.g_value_history[e].append((step, self.g_value[e].item()))


    def update_masks(self, done: list):
        """
        Update global masks based on environment done status.
        Args:
            done: List of boolean flags indicating if each scene is done.
        Returns:
            Local masks for the current step.
        """
        l_masks = torch.FloatTensor([0 if x else 1 for x in done]).to(self.device)
        self.g_masks *= l_masks
        return l_masks


    def init_rollout(self, obs: torch.Tensor, extras: torch.Tensor):
        """
        Initialize rollout storage with initial observation and extras.
        Args:
            obs: Initial observation.
            extras: Initial extra info (orientation).
        """
        self.g_rollouts.obs[0].copy_(obs)
        self.g_rollouts.extras[0].copy_(extras)
        self.g_rollouts.rec_states[0].copy_(self.g_rec_states)


    def act(self, step: int, deterministic: bool = False):
        """
        Sample long-term goal from global policy and compute global goals.
        Args:
            step: Global step index.
            deterministic: Whether to use deterministic actions.
        Returns:
            List of global goals [x, y] for each scene.
        """
        self.g_value, self.g_action, self.g_action_log_prob, self.g_rec_states = \
            self.g_policy.act(
                self.g_rollouts.obs[step],
                self.g_rollouts.rec_states[step],
                self.g_rollouts.masks[step],
                extras=self.g_rollouts.extras[step],
                deterministic=deterministic
            )
        
        action = torch.sigmoid(self.g_action).cpu().numpy()
        global_goals = [[int(a[0] * self.local_w), int(a[1] * self.local_h)]
                        for a in action]
            
        return global_goals


    def insert_rollout(self, obs: torch.Tensor, rewards: torch.Tensor, extras: torch.Tensor):
        """
        Insert samples into global rollout storage and reset masks.
        Args:
            obs: Observation tensor for self.step + 1.
            rewards: Reward tensor for self.step.
            extras: Extra information (orientation) for self.step + 1.
        """
        self.g_rollouts.insert(
            obs, self.g_rec_states, self.g_action, self.g_action_log_prob,
            self.g_value, rewards, self.g_masks, extras
        )
        self.g_masks = torch.ones(self.num_scenes).float().to(self.device)

    def process_rewards(self, infos: list, ep_num: int):
        """
        Process and log rewards and area coverage metrics.
        Args:
            infos: Environment information.
            ep_num: Current episode number.
        Returns:
            Global reward tensor.
        """
        g_reward = torch.from_numpy(np.asarray(
            [infos[env_idx]['exp_reward'] for env_idx in range(self.num_scenes)])
        ).float().to(self.device)

        self.g_process_rewards += g_reward.cpu().numpy()
        g_total_rewards = self.g_process_rewards * (1 - self.g_masks.cpu().numpy())
        self.g_process_rewards *= self.g_masks.cpu().numpy()
        self.logger.log('reward/step mean', np.mean(g_reward.cpu().numpy()))

        if np.sum(g_total_rewards) != 0:
            for tr in g_total_rewards:
                if tr != 0:
                    self.logger.log('reward/ep mean', tr)
               
        exp_ratio = np.asarray([infos[env_idx]['exp_ratio'] for env_idx in range(self.num_scenes)]) 
        self.logger.log('area coverage/step mean', np.mean(exp_ratio))
        self.accumulated_ratio += exp_ratio
        self.all_accumulated_ratios[ep_num].append(self.accumulated_ratio.tolist())
        done_ratio = self.accumulated_ratio * (1 - self.g_masks.cpu().numpy())
        self.accumulated_ratio *= self.g_masks.cpu().numpy()

        if self.args.use_NeRF_mapping:
            uncertainty = np.asarray([infos[env_idx]['uncertainty'] for env_idx in range(self.num_scenes)]) 
            self.uncertainty_list[ep_num].append(uncertainty.tolist())

        if np.sum(done_ratio) != 0:
            for scene_ratio in done_ratio:
                if scene_ratio != 0:
                    self.logger.log('area coverage/ep mean', scene_ratio)
        
        return g_reward

    def train(self):
        """Perform a training update on the global policy."""
        g_next_value = self.g_policy.get_value(
            self.g_rollouts.obs[-1],
            self.g_rollouts.rec_states[-1],
            self.g_rollouts.masks[-1],
            extras=self.g_rollouts.extras[-1]
        ).detach()

        self.g_rollouts.compute_returns(g_next_value, self.args.use_gae,
                                   self.args.gamma, self.args.tau)
        g_value_loss, g_action_loss, g_dist_entropy, approx_kl, clip_fraction = \
            self.g_agent.update(self.g_rollouts)
        
        self.logger.log('loss/value', g_value_loss)
        self.logger.log('loss/action', g_action_loss)
        self.logger.log('loss/dist', g_dist_entropy)
        self.logger.log('info/approx_kl', approx_kl)
        self.logger.log('info/clip_fraction', clip_fraction)
        
        self.g_rollouts.after_update()

    def save_model(self, step: int, dump_dir: str):
        """
        Save the global policy model.
        Args:
            step: Current step index.
            dump_dir: Directory to save the model.
        """
        torch.save(self.g_policy.state_dict(),
                   os.path.join(dump_dir, "periodic_{}.global".format(step)))

    def save_results(self, dump_dir: str):
        """
        Save tracking results to JSON files.
        Args:
            dump_dir: Directory to save the results.
        """
        with open(os.path.join(dump_dir, 'accumulated_ratios.json'), 'w') as f:
            json.dump(self.all_accumulated_ratios, f, indent=4)
        if self.args.use_NeRF_mapping:
            with open(os.path.join(dump_dir, 'uncertainty_history.json'), 'w') as f:
                json.dump(self.uncertainty_list, f, indent=4)
