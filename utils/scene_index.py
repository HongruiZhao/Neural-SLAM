import json
import argparse
from collections import OrderedDict

parser = argparse.ArgumentParser()
parser.add_argument('--scene', type=str, help='Scene name, e.g. Oyens')
args = parser.parse_args()

with open('train_episodes.json') as f:
    episodes = json.load(f)

# Group episodes by scene, preserving file order
scene_episodes = OrderedDict()
for flat_idx, ep in enumerate(episodes):
    scene = ep['scene_id'].split('/')[-1].replace('.glb', '')
    if scene not in scene_episodes:
        scene_episodes[scene] = []
    scene_episodes[scene].append(flat_idx)

if args.scene not in scene_episodes:
    print(f"Scene '{args.scene}' not found.")
    print(f"Available scenes: {list(scene_episodes.keys())}")
else:
    indices = scene_episodes[args.scene]
    print(f"Scene   : {args.scene}")
    print(f"Count   : {len(indices)}")
    print(f"Range   : [{indices[0]}, {indices[-1]}]")
