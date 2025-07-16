import argparse
import yaml
import os 
from tqdm import trange
from pynput import keyboard
import habitat_sim
from habitat_sim.utils.common import d3_40_colors_rgb
import numpy as np
from PIL import Image
import cv2

# local import 
from ramen_mapping import data_loading, get_camera_rays, Mapping


def load_config(path, default_path=None):
    """
    Loads config file.
    Args:
        path (str): path to config file.
        default_path (str, optional): whether to use default path. Defaults to None.
    Returns:
        cfg (dict): config dict.
    """

    def update_recursive(dict1, dict2):
        """
        Update two config dictionaries recursively.
        Args:
            dict1 (dict): first dictionary to be updated.
            dict2 (dict): second dictionary which entries should be used.
        """
        for k, v in dict2.items():
            if k not in dict1:
                dict1[k] = dict()
            if isinstance(v, dict):
                update_recursive(dict1[k], v)
            else:
                dict1[k] = v
                
    # load configuration from file itself
    with open(path, 'r') as f:
        cfg_special = yaml.full_load(f)

    # check if we should inherit from a config
    inherit_from = cfg_special.get('inherit_from')

    # if yes, load this config first as default
    # if no, use the default_path
    if inherit_from is not None:
        cfg = load_config(inherit_from, default_path)
    elif default_path is not None:
        with open(default_path, 'r') as f:
            cfg = yaml.full_load(f)
    else:
        cfg = dict()

    # include main configuration
    update_recursive(cfg, cfg_special)

    return cfg


def habitat_sim_cfg(cfg):
    sim_cfg = habitat_sim.SimulatorConfiguration()
    sim_cfg.gpu_device_id = 0
    sim_cfg.scene_id = os.path.join( cfg["scene"]["data_path"], cfg["scene"]["scene"] )
    #sim_cfg.scene_dataset_config_file = os.path.join( cfg["scene"]["data_path"], cfg["scene"]["scene_dataset"] )
    sim_cfg.enable_physics = cfg["scene"]["enable_physics"]

    # Note: all sensors must have the same resolution
    sensor_specs = []

    color_sensor_spec = habitat_sim.CameraSensorSpec()
    color_sensor_spec.uuid = "color_sensor"
    color_sensor_spec.sensor_type = habitat_sim.SensorType.COLOR
    color_sensor_spec.resolution = [cfg["agent"]["height"], cfg["agent"]["width"]]
    color_sensor_spec.position = [0.0, cfg["agent"]["sensor_height"], 0.0]
    color_sensor_spec.sensor_subtype = habitat_sim.SensorSubType.PINHOLE
    sensor_specs.append(color_sensor_spec)

    if cfg["agent"]["depth_sensor"]:
        depth_sensor_spec = habitat_sim.CameraSensorSpec()
        depth_sensor_spec.uuid = "depth_sensor"
        depth_sensor_spec.sensor_type = habitat_sim.SensorType.DEPTH
        depth_sensor_spec.resolution = [cfg["agent"]["height"], cfg["agent"]["width"]]
        depth_sensor_spec.position = [0.0, cfg["agent"]["sensor_height"], 0.0]
        depth_sensor_spec.sensor_subtype = habitat_sim.SensorSubType.PINHOLE
        sensor_specs.append(depth_sensor_spec)

    if cfg["agent"]["semantic_sensor"]:
        semantic_sensor_spec = habitat_sim.CameraSensorSpec()
        semantic_sensor_spec.uuid = "semantic_sensor"
        semantic_sensor_spec.sensor_type = habitat_sim.SensorType.SEMANTIC
        semantic_sensor_spec.resolution = [cfg["agent"]["height"], cfg["agent"]["width"]]
        semantic_sensor_spec.position = [0.0, cfg["agent"]["sensor_height"], 0.0]
        semantic_sensor_spec.sensor_subtype = habitat_sim.SensorSubType.PINHOLE
        sensor_specs.append(semantic_sensor_spec)

    # Here you can specify the amount of displacement in a forward action and the turn angle
    agent_cfg = habitat_sim.agent.AgentConfiguration()
    agent_cfg.sensor_specifications = sensor_specs
    agent_cfg.action_space = {
        "move_forward": habitat_sim.agent.ActionSpec(
            "move_forward", habitat_sim.agent.ActuationSpec(amount=cfg["action"]["move_forward"])
        ),
        "turn_left": habitat_sim.agent.ActionSpec(
            "turn_left", habitat_sim.agent.ActuationSpec(amount=cfg["action"]["turn"])
        ),
        "turn_right": habitat_sim.agent.ActionSpec(
            "turn_right", habitat_sim.agent.ActuationSpec(amount=cfg["action"]["turn"])
        ),
    }

    return habitat_sim.Configuration(sim_cfg, [agent_cfg])


def display_sample(rgb_obs, semantic_obs=np.array([]), depth_obs=np.array([])):
    
    rgb_img = cv2.cvtColor(rgb_obs, cv2.COLOR_RGBA2BGR)
    arr = [rgb_img]

    if semantic_obs.size != 0:
        semantic_img = Image.new("P", (semantic_obs.shape[1], semantic_obs.shape[0]))
        semantic_img.putpalette(d3_40_colors_rgb.flatten())
        semantic_img.putdata((semantic_obs.flatten() % 40).astype(np.uint8))
        semantic_img = semantic_img.convert("RGB")
        semantic_img = np.array(semantic_img) # Convert PIL to NumPy
        semantic_img = cv2.cvtColor(semantic_img, cv2.COLOR_RGB2BGR) # Convert to BGR for OpenCV
        arr.append(semantic_img)
    if depth_obs.size != 0:
        depth_img = cv2.cvtColor((depth_obs / 10 * 255).astype(np.uint8), cv2.COLOR_GRAY2BGR)
        depth_img = cv2.applyColorMap(depth_img, cv2.COLORMAP_JET)
        arr.append(depth_img)

    combined_images = np.concatenate(arr, axis=1)
    cv2.imshow("Observations", combined_images)
    cv2.waitKey(1) # wait time in ms


def on_press(key, injected):
    global cmd 
    try:
        if key.char == 'w':
            cmd = 'move_forward'
        elif key.char== 'a':
            cmd = 'turn_left'
        elif key.char == 'd':
            cmd = 'turn_right'
        else:
            print('  wrong input')
            return 
        print('  ' + cmd)

    except AttributeError:
        print('special key {} pressed'.format(key))


def get_camera_intrinsics(sim, sensor_name):
    """
        https://github.com/facebookresearch/habitat-sim/issues/2439
    """
    # Get render camera
    render_camera = sim._sensors[sensor_name]._sensor_object.render_camera

    # Get projection matrix
    projection_matrix = render_camera.projection_matrix

    # Get resolution
    viewport_size = render_camera.viewport

    # Intrinsic calculation
    fx = projection_matrix[0, 0] * viewport_size[0] / 2.0
    fy = projection_matrix[1, 1] * viewport_size[1] / 2.0
    cx = (projection_matrix[2, 0] + 1.0) * viewport_size[0] / 2.0
    cy = (projection_matrix[2, 1] + 1.0) * viewport_size[1] / 2.0

    return fx, fy, cx, cy 


def main(habitat_cfg, mapping_cfg):
    # set up habitat sim 
    sim = habitat_sim.Simulator(habitat_sim_cfg(habitat_cfg))
    max_frames = habitat_cfg['sim']['max_frames']
    agent = sim.initialize_agent(habitat_cfg["agent"]["default_agent"])
    agent_state = habitat_sim.AgentState()
    agent_state.position = np.array([0, 0.0, 0.0])  # world space
    agent.set_state(agent_state)    

    
    # set up mapper
    H, W = habitat_cfg["agent"]["height"], habitat_cfg["agent"]["width"]
    fx, fy, cx, cy = get_camera_intrinsics(sim, 'depth_sensor')
    num_rays_to_save = int(H*W*mapping_cfg['mapping']['n_pixels'])
    dataset_info = {'num_frames':max_frames, 'num_rays_to_save':num_rays_to_save, 'H':H, 'W':W }
    rays_d = get_camera_rays(H, W, fx, fy, cx, cy)
    mapper = Mapping(mapping_cfg, id=0, dataset_info=dataset_info)
  

    # set up keyboard 
    # in a non-blocking fashion:
    global cmd 
    cmd = None
    listener = keyboard.Listener(on_press=on_press)
    listener.start()

    for i in trange(max_frames):
        
        while cmd == None:
            pass

        observations = sim.step(cmd)
        agent = sim.get_agent(0) # c2w
        agent_pos = agent.state.sensor_states['depth_sensor'].position
        print(f'pos = {agent_pos}')
        agent_q = agent.state.sensor_states['depth_sensor'].rotation # quat (w,x,y,z)
        cmd = None

        rgb = observations["color_sensor"]
        if habitat_cfg["agent"]["semantic_sensor"]:
            semantic = observations["semantic_sensor"]
        else: 
            semantic = np.array([])
        if habitat_cfg["agent"]["depth_sensor"]:
            depth = observations["depth_sensor"]
        else: 
            depth = np.array([])

        display_sample(rgb, semantic, depth)
        batch = data_loading(rgb, depth, agent_pos, agent_q, step=i, rays_d=rays_d)
        mapper.run(i, batch)

            

    



if __name__ == '__main__':

    print('Start running...')
    parser = argparse.ArgumentParser(
        description='Arguments for running the NICE-SLAM/iMAP*.'
    )
    parser.add_argument('--habitat_config', type=str, default='./configs/habitat.yaml', help='Path to habitat config file.')
    parser.add_argument('--mapping_config', type=str, default='./configs/mapping.yaml', help='Path to maping config file.')

    args = parser.parse_args()

    habitat_cfg = load_config(args.habitat_config)
    mapping_cfg = load_config(args.mapping_config) 

    main(habitat_cfg, mapping_cfg)

