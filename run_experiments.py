import os
import shutil
import subprocess
import yaml
import multiprocessing
from tqdm import tqdm

MAPPING_CONFIG_PATH = 'env/habitat/configs/mapping.yaml'

def load_global_config(path):
    """Parses a global config file (key = value) into a dict."""
    config = {}
    with open(path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            if '=' in line:
                key, value = line.split('=', 1)
                config[key.strip()] = value.strip()
    return config


def write_global_config(config, path):
    """Writes a dict to a global config file."""
    with open(path, 'w') as f:
        for key, value in config.items():
            # If value contains comment, keep it
            f.write(f"{key} = {value}\n")

def update_recursive(d, u):
    """Recursive update for mapping overrides."""
    for k, v in u.items():
        if isinstance(v, dict):
            if k not in d:
                d[k] = {}
            d[k] = update_recursive(d.get(k, {}), v)
        else:
            d[k] = v
    return d

def run_experiment(exp_config):
    
    global_cfg = load_global_config(exp_config['global_config_path'])
    global_cfg.update(exp_config.get('global_overrides', {}))
    
    exp_name = global_cfg.get('exp_name', 'unnamed')
    # Unique temp config per experiment to allow parallel execution
    temp_global_config_path = f'temp_global_config_{exp_name}.txt'
    write_global_config(global_cfg, temp_global_config_path)
    
    base_mapping_path = exp_config.get('mapping_config_path', MAPPING_CONFIG_PATH)
    with open(base_mapping_path, 'r') as f:
        mapping_cfg = yaml.full_load(f)
    
    update_recursive(mapping_cfg, exp_config.get('mapping_overrides', {}))
    
    # Unique mapping config per experiment
    temp_mapping_config_path = f'temp_mapping_config_{exp_name}.yaml'
    
    try:
        with open(temp_mapping_config_path, 'w') as f:
            yaml.dump(mapping_cfg, f)
            
        print(f"Running experiment: {exp_name}")
        
        # Pass the unique mapping config to main.py. run in silent mode
        cmd = f"python main.py --config {temp_global_config_path} --mapping_config {temp_mapping_config_path}"
        subprocess.run(cmd, shell=True, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        
        dump_location = global_cfg.get('dump_location', './tmp/').strip()
        
        log_dir = os.path.join(dump_location, 'models', exp_name)
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)
            
        # Copy configs
        shutil.copy2(temp_global_config_path, os.path.join(log_dir, 'global_config.txt'))
        shutil.copy2(temp_mapping_config_path, os.path.join(log_dir, 'mapping.yaml'))
        
        print(f"Configs saved to {log_dir}")
        
    except subprocess.CalledProcessError as e:
        print(f"Experiment {exp_name} failed with error: {e}")
    except Exception as e:
        print(f"An error occurred in {exp_name}: {e}")
    finally:
        # Remove temp configs
        if os.path.exists(temp_global_config_path):
            os.remove(temp_global_config_path)
        if os.path.exists(temp_mapping_config_path):
            os.remove(temp_mapping_config_path)


def main():
    base_global_config = 'configs/eval_NSLAM_Annawan.txt'
    base_mapping_config = 'env/habitat/configs/mapping.yaml'
    eval_name = 'March18_evalNSLAM_Annawan'
    exp_name = 'March18_trainNSLAM_Annawan'
    #step = [400000, 600000, 800000, 1000000, 1200000, 1240000]

    step = [1240000]

    exp_names = [ f'{eval_name}_{s}' for s in step ]
    loag_global = [f'results/dump/{exp_name}/periodic_{s}.global' for s in step]
    experiments = []
    for i in range(len(exp_names)):
        experiments.append({
            "global_config_path": base_global_config,
            "mapping_config_path": base_mapping_config,
            "global_overrides": {
                "split": "train",
                "exp_name": exp_names[i],
                "eval_scene_id": 400000,
                "load_global": loag_global[i],
                "max_episode_length": 500,
            },
            "mapping_overrides": {
                "grid": { "uncertainty": 'ensemble', 'ensemble_size': 3} ,
                "mapping":{'replay': True, 'iters':50},
                "mesh":{'vis':500},
            },
        })

    # Run in parallel
    num_parallel = len(experiments)
    print(f"Starting {num_parallel} experiments in parallel...")
    with multiprocessing.Pool(processes=num_parallel) as pool:
        for _ in tqdm(pool.imap_unordered(run_experiment, experiments), total=len(experiments)):
            pass

    print("\nAll experiments completed.")

if __name__ == "__main__":
    main()
