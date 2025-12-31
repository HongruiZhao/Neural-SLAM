import argparse
import json
import os
import copy
import re
import numpy as np
import config
import sys
import subprocess
import glob

def run_command(command):
    """Runs a command and returns its stdout, printing errors if any."""
    try:
        result = subprocess.run(command, shell=True, capture_output=True, text=True, check=True)
        return result.stdout
    except subprocess.CalledProcessError as e:
        print(f"Error running command: {command}")
        print(f"Return code: {e.returncode}")
        print(f"Stdout: {e.stdout}")
        print(f"Stderr: {e.stderr}")
        return None


def extract_metrics_from_output(output):
    """Extracts metrics from the stdout of eval_recon.py."""
    if output is None:
        return None, None, None
        
    accuracy_match = re.search(r"accuracy: (\d+\.?\d*)", output)
    completion_match = re.search(r"completion: (\d+\.?\d*)", output)
    completion_ratio_match = re.search(r"completion ratio: (\d+\.?\d*)", output)

    acc = float(accuracy_match.group(1)) if accuracy_match else None
    comp = float(completion_match.group(1)) if completion_match else None
    comp_ratio = float(completion_ratio_match.group(1)) if completion_ratio_match else None

    return acc, comp, comp_ratio


def find_last_mesh(agent_path):
    """Finds the mesh file with the highest track number."""
    mesh_files = glob.glob(os.path.join(agent_path, 'mesh_track*.ply'))
    if not mesh_files:
        return None

    latest_file = None
    max_track_num = -1
    p = re.compile(r'mesh_track(\d+)(?:_cull_occlusion|_cull_virt_cams)?\.ply')
    for f in mesh_files:
        basename = os.path.basename(f)

        if '_cull' in basename:
            return f
        match = p.match(basename)
        if match:
            track_num = int(match.group(1))
            if track_num > max_track_num:
                max_track_num = track_num
                latest_file = f
    return latest_file


def main():
    parser = argparse.ArgumentParser(description='Automatic evaluation script for LENA.')
    parser.add_argument('--config', type=str, default='eval/eval_basic.yaml',
                        help='Path to the experiment configuration file.')
    parser.add_argument('--skip_cull', action='store_true',
                        help='skip culled mesh')
    args = parser.parse_args()
    
    exp_cfg = config.load_config(args.config)
    mapping_cfg_path = exp_cfg['mapping_cfg']
    for exp_name, episode, thread in \
        zip(exp_cfg['exp_names'], exp_cfg['episodes'], exp_cfg['threads']):
            base_path = os.path.join('results/mapping', f'{exp_name}_ep{episode}', f'agent_{thread}')
            mesh_files = glob.glob(os.path.join(base_path, 'mesh*.ply'))
            scene_name = mesh_files[-1].split('/')[-1].split('_')[1]
            gt_mesh = os.path.join(exp_cfg['gt_path'], f'{scene_name}.glb')

            for input_mesh in mesh_files:
                step = input_mesh.split('/')[-1].split('_')[-1].split('.')[0]
                ckpt_path =  os.path.join(base_path, f'checkpoint_{step}.pt')

                # 1. Cull Mesh
                if ('_cull' in input_mesh) or args.skip_cull:
                    print(f"  cull mesh exists: {input_mesh}")
                    rec_mesh_culled = input_mesh
                else:
                    print(f"  Processing mesh: {input_mesh}")
                    cull_cmd = (
                            f"python ./eval/cull_mesh.py --config {mapping_cfg_path} --input_mesh {input_mesh} "
                            f"--remove_occlusion --ckpt_path {ckpt_path}"
                        )
                    rec_mesh_culled = input_mesh.replace('.ply', '_cull_occlusion.ply')
                    run_command(cull_cmd)
                # 2. Evaluate Reconstruction
                eval_cmd = (f"python ./eval/eval_recon.py --rec_mesh {rec_mesh_culled} --gt_mesh {gt_mesh} ")
                eval_output = run_command(eval_cmd)

                # 3. Extract and store metrics
                acc, comp, comp_ratio = extract_metrics_from_output(eval_output)



if __name__ == '__main__':
    main()
