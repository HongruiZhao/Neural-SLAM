import time
from collections import deque

import os

os.environ["OMP_NUM_THREADS"] = "1"
import numpy as np

import gym
import logging
from arguments import get_args
from env import make_vec_envs

import cv2

import sys
import matplotlib
import torch

from env.habitat import make_env_fn, construct_envs

args = get_args()

np.random.seed(args.seed)
torch.manual_seed(args.seed)

if args.cuda:
    torch.cuda.manual_seed(args.seed)


# --- Config ---
CONFIG_PATH = "/u/jqw4ax/exp_thread/Neural-SLAM/env/habitat/configs/gibson_half_step.yaml"
# Map keys to Habitat actions
KEY_ACTIONS = {
    ord('w'): "move_forward",
    ord('a'): "turn_left",
    ord('d'): "turn_right",
    ord('s'): "stop",
    27: "EXIT",  # ESC
}

def display_obs(obs):
    """Render RGB frame."""
    print(obs)
    # if "rgb" not in obs:
    #     print("No RGB channel found in observation.")
    #     return
    # rgb = obs["rgb"][:, :, ::-1]  # RGB→BGR for OpenCV
    rgb = np.transpose(obs, (1, 2, 0))[:, :, ::-1]
    print(obs.shape)
    cv2.imshow("Habitat Debug View", rgb)


def print_status(info):
    """Print minimal status info."""
    pos = info.get("agent_position", None)
    coll = info.get("collisions", {}).get("is_collision", False)
    if pos is not None:
        pos_str = np.round(pos, 2)
    else:
        pos_str = "unknown"
    print(f"Position: {pos_str} | Collision: {coll}")

def main():
    args.output_debug_env = 1
    env = construct_envs(args)
    
    print("ENV!!!!!!!!! ", env)

    env.reset()
    env.reset()
    obs, _, done, info = env.step({"action": "stop"})
    display_obs(obs)
    obs = env.reset()

    print("Controls: [W] forward, [A] left, [D] right, [S] stop, [ESC] exit")

    while True:
        key = cv2.waitKey(0) & 0xFF
        if key not in KEY_ACTIONS:
            continue

        action = KEY_ACTIONS[key]

        if action == "EXIT":
            print("Exiting...")
            break

        print("Done? ", done)
        obs, _, done, info = env.step({"action": action})
        display_obs(obs)
        print_status(info)

        if done:
            print("Episode done. Resetting.")
            obs = env.reset()
            display_obs(obs)

    env.close()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()