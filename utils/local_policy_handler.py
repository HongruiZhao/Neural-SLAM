import torch
import numpy as np
from model import DdppoPolicy

class LocalPolicyHandler:
    """
    Class handling the initialization and execution of the local policy.
    """
    def __init__(self, args, num_scenes: int, device: torch.device):
        """
        Initialize the LocalPolicyHandler.
        Args:
            args: Command line arguments.
            num_scenes: Number of parallel environments.
            device: Torch device.
        """
        self.args = args
        self.num_scenes = num_scenes
        self.device = device
        
        self.l_policy = DdppoPolicy(path=args.use_DD_PPO, num_scenes=num_scenes).to(device)
        if hasattr(torch, 'compile'):
            self.l_policy = torch.compile(self.l_policy)


    def reset(self):
        """Reset the local policy state."""
        self.l_policy.reset()


    def plan(self, infos, global_goals, planner_pose_inputs, g_masks, step):
        """
        Plan the next local action.
        Args:
            infos: Environment information.
            global_goals: Current global goals.
            planner_pose_inputs: Inputs for the planner.
            g_masks: Global masks.
            step: Current step index.
        Returns:
            Local action tensor.
        """
        depth = np.stack([infos[e]['depth'] for e in range(self.num_scenes)])
        l_action = self.l_policy.plan(depth, 
                                     global_goals, planner_pose_inputs, 
                                     g_masks, step, 
                                     self.args.map_resolution, self.device)
        return l_action
