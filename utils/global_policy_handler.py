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
        # Sticky per-env termination flag for the current outer episode.
        # Cleared in reset_episode(); set True the first time an env's
        # episode ends so subsequent post-termination rewards are ignored.
        self.episode_done = np.zeros(num_scenes, dtype=bool)

        self.finished_reward_snapshots = np.zeros(num_scenes)
        self.finished_ratio_snapshots = np.zeros(num_scenes)
        self.finished_step_snapshots = np.full(num_scenes, args.max_episode_length, dtype=np.int64)


    def load_model(self, path: str):
        """
        Load global policy model from path.
        Args:
            path: Path to the model state dict.
        """
        if path != "0":
            print("Loading global {}".format(path))
            state_dict = torch.load(path, map_location=lambda storage, loc: storage)
            # torch.compile() wraps the model with _orig_mod. prefix; remap if needed
            if hasattr(self.g_policy, '_orig_mod'):
                first_key = next(iter(state_dict))
                if not first_key.startswith('_orig_mod.'):
                    state_dict = {'_orig_mod.' + k: v for k, v in state_dict.items()}
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
        self.episode_done.fill(False)
        self.finished_reward_snapshots.fill(0.)
        self.finished_ratio_snapshots.fill(0.)
        self.finished_step_snapshots.fill(self.args.max_episode_length)


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


    def get_global_goals_m(self, global_goals, planner_pose_inputs):
        """
        Transform global pixel goals to global coordinates in meters.
        Args:
            global_goals: List of [x, y] pixel coordinates in local window.
            planner_pose_inputs: Matrix containing local map boundaries (gx1, gy1 at indices 3, 5).
        Returns:
            List of [x, y] coordinates in meters.
        """
        global_goals = np.array(global_goals)
        # goal_x = (pixel_x + gx1) * map_resolution / 100
        goal_x = (global_goals[:, 0] + planner_pose_inputs[:, 3]) * self.args.map_resolution / 100.0
        # goal_y = (pixel_y + gy1) * map_resolution / 100
        goal_y = (global_goals[:, 1] + planner_pose_inputs[:, 5]) * self.args.map_resolution / 100.0
        
        global_goals_m = np.stack([goal_x, goal_y], axis=1).tolist()
        return global_goals_m


    def insert_rollout(self, obs: torch.Tensor, rewards: torch.Tensor, extras: torch.Tensor):
        """
        Insert samples into global rollout storage and reset masks.
        Args:
            obs: Observation tensor for self.step + 1.
            rewards: Reward tensor for self.step.
            extras: Extra information (orientation) for self.step + 1.
        """

        if self.args.uncert_only:
            obs[:,0] = obs[:,1]
            obs[:,4] = obs[:,5]


        self.g_rollouts.insert(
            obs, self.g_rec_states, self.g_action, self.g_action_log_prob,
            self.g_value, rewards, self.g_masks, extras
        )
        self.g_masks = torch.ones(self.num_scenes).float().to(self.device)

    def process_rewards(self, infos: list, ep_num: int, step: int):
        """
        Process and log rewards and area coverage metrics.
        Args:
            infos: Environment information.
            ep_num: Current episode number.
            step: Current per-episode step index (used to record finish step).
        Returns:
            Global reward tensor.
        """

        g_reward_np = np.asarray(
            [infos[env_idx]['exp_reward'] for env_idx in range(self.num_scenes)]
        )
        g_reward = torch.from_numpy(g_reward_np).float().to(self.device)

        active_mask = ~self.episode_done
        g_masks_np = self.g_masks.cpu().numpy()
        just_finished = active_mask & (g_masks_np == 0)

        # Per-step reward, averaged over envs not yet terminated.
        if active_mask.any():
            self.logger.log(
                'reward/reward per global step(avg over envs)',
                g_reward_np[active_mask].mean(),
            )

        # Per-step reward components, averaged over envs not yet terminated.
        for key in ['uncert_reward', 'area_reward', 'dist_penalty', 'time_penalty', 'finish_bonus']:
            component_values = []
            for env_idx in range(self.num_scenes):
                if not active_mask[env_idx]:
                    continue
                comps = infos[env_idx].get('reward_components')
                if comps is not None and key in comps:
                    component_values.append(comps[key])
            if len(component_values) > 0:
                if key == 'finish_bonus':
                    self.logger.log(
                        f'reward components per global step (avg over envs)/{key}',
                        np.sum(component_values),
                    )
                else:
                    self.logger.log(
                        f'reward components per global step (avg over envs)/{key}',
                        np.mean(component_values),
                    )

        # Accumulate episode totals only for envs whose rewards are still valid.
        self.g_process_rewards[active_mask] += g_reward_np[active_mask]

        # Area coverage.
        exp_ratio = np.asarray(
            [infos[env_idx]['exp_ratio'] for env_idx in range(self.num_scenes)]
        )
        if active_mask.any():
            self.logger.log(
                'area coverage/step mean (avg over envs)',
                exp_ratio[active_mask].mean(),
            )
        self.accumulated_ratio[active_mask] += exp_ratio[active_mask]
        self.all_accumulated_ratios[ep_num].append(self.accumulated_ratio.tolist())

        if self.args.use_NeRF_mapping:
            uncertainty = np.asarray(
                [infos[env_idx]['uncertainty'] for env_idx in range(self.num_scenes)]
            )
            self.uncertainty_list[ep_num].append(uncertainty.tolist())

        # Snapshot per-env totals for envs that JUST finished, so the
        # end-of-episode summary can combine these with still-active envs'
        # current accumulators.
        if just_finished.any():
            self.finished_reward_snapshots[just_finished] = self.g_process_rewards[just_finished]
            self.finished_ratio_snapshots[just_finished] = self.accumulated_ratio[just_finished]
            self.finished_step_snapshots[just_finished] = step

        # Mark just-finished envs as done so subsequent global steps skip them.
        self.episode_done |= just_finished

        return g_reward


    def log_episode_summary(self):

        rewards = np.where(
            self.episode_done, self.finished_reward_snapshots, self.g_process_rewards
        )
        ratios = np.where(
            self.episode_done, self.finished_ratio_snapshots, self.accumulated_ratio
        )
        self.logger.log(
            'reward/ep mean accumulated reward (avg over envs)', rewards.mean()
        )
        self.logger.log(
            'reward/ep mean finish step (avg over envs)',
            self.finished_step_snapshots.mean(),
        )
        self.logger.log('area coverage/ep mean', ratios.mean())


    def train(self):
        """Perform a training update on the global policy."""
        print("Training global policy...")
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
