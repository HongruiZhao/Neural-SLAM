import argparse
import json
import os
import re
import config
import subprocess
import glob
from tqdm import tqdm 
import itertools
import re
from multiprocessing import Pool


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


class cull_and_evaluate:
    def __init__(self,base_path, culled_mesh_files, skip_cull, 
                 mapping_cfg_path, gt_mesh,):
        self.base_path = base_path 
        self.culled_mesh_files = culled_mesh_files
        self.skip_cull = skip_cull
        self.mapping_cfg_path = mapping_cfg_path
        self.gt_mesh = gt_mesh

    def process(self,input_mesh):
        step = input_mesh.split('/')[-1].split('_')[-1].split('.')[0]
        ckpt_path =  os.path.join(self.base_path, f'checkpoint_{step}.pt')

        # 1. Cull Mesh
        cull_exist = (input_mesh).split('.')[0] + '_cull_occlusion' + '.ply' in self.culled_mesh_files
        if cull_exist or self.skip_cull:
            rec_mesh_culled = (input_mesh).split('.')[0] + '_cull_occlusion' + '.ply'
        else:
            cull_cmd = (
                    f"python ./eval/cull_mesh.py --config {self.mapping_cfg_path} --input_mesh {input_mesh} "
                    f"--remove_occlusion --ckpt_path {ckpt_path}"
                )
            rec_mesh_culled = input_mesh.replace('.ply', '_cull_occlusion.ply')
            run_command(cull_cmd)
        # 2. Evaluate Reconstruction
        eval_cmd = (f"python ./eval/eval_recon.py --rec_mesh {rec_mesh_culled} \
                    --gt_mesh {self.gt_mesh} --ckpt_path {ckpt_path} ")
        eval_output = run_command(eval_cmd)

        # 3. Extract and store metrics
        acc, comp, comp_ratio = extract_metrics_from_output(eval_output)
        return {
            "acc": acc,
            "comp": comp,
            "comp_ratio": comp_ratio,
            "mesh_name": input_mesh.split('/')[2:]
        }


def main():
    parser = argparse.ArgumentParser(description='Automatic evaluation script for LENA.')
    parser.add_argument('--config', type=str, default='eval/eval_basic.yaml',
                        help='Path to the experiment configuration file.')
    parser.add_argument('--skip_cull', action='store_true',
                        help='skip culled mesh')
    args = parser.parse_args()
    
    exp_cfg = config.load_config(args.config)
    mapping_cfg_path = exp_cfg['mapping_cfg']
    num_processes = exp_cfg['num_processes']
    all_exps = list(itertools.product(exp_cfg['exp_names'], exp_cfg['episodes'], exp_cfg['threads']))
    evaluation_results = {"acc":[], "comp":[], "comp_ratio":[], "mesh_name":[]}

    for exp in tqdm(all_exps, desc='outer'):
            base_path = os.path.join('results/mapping', f'{exp[0]}_ep{exp[1]}', f'agent_{exp[2]}')
            mesh_files = glob.glob(os.path.join(base_path, 'mesh*.ply'))
            culled_mesh_files = glob.glob(os.path.join(base_path, 'mesh*_cull_occlusion.ply'))
            mesh_files = [mesh for mesh in mesh_files if mesh not in culled_mesh_files]
            mesh_files.sort(key=lambda x: int(x.split('/')[-1].split('_')[-1].split('.')[0]) )
            scene_name = mesh_files[-1].split('/')[-1].split('_')[1]
            gt_mesh = os.path.join(exp_cfg['gt_path'], f'{scene_name}.glb')

            # Parallel culling & evaluation
            cull_and_evaluate_obj = cull_and_evaluate(base_path, culled_mesh_files, args.skip_cull, 
                                                      mapping_cfg_path, gt_mesh)
            
            with Pool(processes=num_processes) as pool:
                results = list(tqdm(
                    pool.imap(cull_and_evaluate_obj.process, mesh_files, chunksize=5), 
                    total=len(mesh_files),
                    desc="Processing meshes"
                ))

            for res in results:
                evaluation_results["acc"].append(res["acc"])
                evaluation_results["comp"].append(res["comp"])
                evaluation_results["comp_ratio"].append(res["comp_ratio"])
                evaluation_results["mesh_name"].append(res["mesh_name"])

    with open('./eval/evaluation_results.json', 'w') as f:
        json.dump(evaluation_results, f, indent=4)


if __name__ == '__main__':
    main()
