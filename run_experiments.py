import os
import shutil
import subprocess
import yaml
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


def run_experiment(exp_config):
    
    global_cfg = load_global_config(exp_config['global_config_path'])
    global_cfg.update(exp_config.get('global_overrides', {}))
    
    temp_global_config_path = 'temp_global_config.txt'
    write_global_config(global_cfg, temp_global_config_path)
    
    base_mapping_path = exp_config.get('mapping_config_path', MAPPING_CONFIG_PATH)
    
    with open(base_mapping_path, 'r') as f:
        mapping_cfg = yaml.full_load(f)
    
    # Recursive update for mapping overrides
    def update_recursive(d, u):
        for k, v in u.items():
            if isinstance(v, dict):
                d[k] = update_recursive(d.get(k, {}), v)
            else:
                d[k] = v
        return d

    update_recursive(mapping_cfg, exp_config.get('mapping_overrides', {}))
    
    # Backup existing mapping config if we are about to overwrite it
    backup_mapping_path = MAPPING_CONFIG_PATH + '.bak'
    if os.path.exists(MAPPING_CONFIG_PATH):
        shutil.copy2(MAPPING_CONFIG_PATH, backup_mapping_path)
    
    try:
        with open(MAPPING_CONFIG_PATH, 'w') as f:
            yaml.dump(mapping_cfg, f)
            
        print(f"Running experiment: {global_cfg.get('exp_name', 'unnamed')}")
        print(f"Global Config: {temp_global_config_path}")
        print(f"Mapping Config: {MAPPING_CONFIG_PATH}")
        
        cmd = f"python main.py --config {temp_global_config_path}"
        subprocess.run(cmd, shell=True, check=True)
        
        dump_location = global_cfg.get('dump_location', './tmp/').strip()
        exp_name = global_cfg.get('exp_name', 'exp1').strip()

        
        log_dir = os.path.join(dump_location, 'models', exp_name)
        if not os.path.exists(log_dir):
            os.makedirs(log_dir) # Should have been created by main.py, but just in case
            
        # Copy configs
        shutil.copy2(temp_global_config_path, os.path.join(log_dir, 'global_config.txt'))
        shutil.copy2(MAPPING_CONFIG_PATH, os.path.join(log_dir, 'mapping.yaml'))
        
        print(f"Configs saved to {log_dir}")
        
    except subprocess.CalledProcessError as e:
        print(f"Experiment failed with error: {e}")
    except Exception as e:
        print(f"An error occurred: {e}")
    finally:
        # Restore Mapping Config
        if os.path.exists(backup_mapping_path):
            shutil.copy2(backup_mapping_path, MAPPING_CONFIG_PATH)
            os.remove(backup_mapping_path)
        
        # Remove temp global config
        if os.path.exists(temp_global_config_path):
            os.remove(temp_global_config_path)

def main():
    
    base_global_config = 'configs/eval_NSLAM.txt'
    base_mapping_config = 'env/habitat/configs/mapping.yaml'
    
    exp_names = ['Feb4_Cantwell_yesReplay_NARUTO_iter100',
                 'Feb4_Eudora_yesReplay_Ensemble_iter100',]
    eval_scen_ids = ['no', '392']
    uncertainty = ['NARUTO', 'ensemble']
    experiments = []
    for i in range(len(exp_names)):
        experiments.append({
            "global_config_path": base_global_config,
            "mapping_config_path": base_mapping_config,
            "global_overrides": {
                "exp_name": exp_names[i],
                "eval_scene_id": eval_scen_ids[i],
            },
            "mapping_overrides": {
                "grid": { "uncertainty": uncertainty[i]} ,
                "mesh": { "vis": 5000000000000},
                "mapping":{'replay': True, 'iters':100},
            },
        })

    for exp in tqdm(experiments):
        run_experiment(exp)
    print("\nAll experiments completed.")

if __name__ == "__main__":
    main()
