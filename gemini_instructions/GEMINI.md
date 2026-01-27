# README

# Install

- First create the conda environmnet. *lena* stands for “learn to perform neural active mapping”. `python=3.9` is a must because `habitat` does not work with newer version python

```bash
conda create -n lena python=3.9 cmake=3.14.0
conda activate lena
```

## Install habitat sim from source 
* The instruction below largely follows [this](https://github.com/facebookresearch/habitat-sim/blob/main/BUILD_FROM_SOURCE.md).  
* Clone the repository
```bash 
git clone --branch stable https://github.com/facebookresearch/habitat-sim.git
cd habitat-sim
```
* Install dependencies 
```bash 
pip install -r requirements.txt
```
* Install libraries needed for building. `libgl1-mesa-glx` in the original instruction is for the older ubuntu version. Replace it with `libgl1` and `libglx-mesa0` instead following [this](https://askubuntu.com/questions/1517352/issues-installing-libgl1-mesa-glx). 
```bash 
sudo apt-get update
sudo apt-get install -y --no-install-recommends libjpeg-dev libglm-dev libgl1 libglx-mesa0 libegl1-mesa-dev mesa-utils xorg-dev freeglut3-dev
```
* Build Habitat-sim with CUDA. Before doing so you will need to make sure [cuda toolkit](https://developer.nvidia.com/cuda-12-6-0-download-archive?target_os=Linux&target_arch=x86_64&Distribution=Ubuntu&target_version=22.04) has been installed. 
```bash 
python setup.py install --with-cuda
```
* For headless systems (i.e. without an attached display, e.g. in a cluster) and multiple GPU systems
```bash 
python setup.py install --headless --with-cuda
```

## Install habitat lab and habitat baseline
* The instruction below largely follows [this](https://github.com/facebookresearch/habitat-lab).
* Install Habitat-lab 
```bash
cd ..
git clone --branch stable https://github.com/facebookresearch/habitat-lab.git
cd habitat-lab
pip install -e habitat-lab  # install habitat_lab
```
* Install habitat_baselines along with all additional requirements. This would also install pytroch and torchvision.
```bash 
pip install -e habitat-baselines  
```

## Install packages for lena
- Now install the rest of the python packages.

```bash
cd ..
git clone https://github.com/HongruiZhao/Neural-SLAM.git
cd Neural-SLAM
pip install -r requirements.txt
```

- Now install tinycudann and pytorch3D for CoSLAM neural implicit mapping
```bash 
conda install -c conda-forge libstdcxx-ng=13 # get GLIBCXX_3.4.32 for tinycudann
pip install git+https://github.com/NVlabs/tiny-cuda-nn/#subdirectory=bindings/torch
pip install "git+https://github.com/facebookresearch/pytorch3d.git"
```

# Download Gibson dataset 
* The paths to the datasets need to be modified in `Neural-SLAM/env/habitat/configs/gibson.yaml`. You should modify `data_path` (for the task dataset) and `scenes_dir` (for the gibson scenes) under `dataset`.
* The Gibson dataset for use with Habitat can be downloaded by agreeing to the terms of use in the Gibson repository. All training and validation scenes of the Gibson dataset for use with Habitat Sim can be downloaded following the commands below. More instructions [here](https://github.com/facebookresearch/habitat-sim/blob/main/DATASETS.md#gibson-and-3dscenegraph-datasets).
```bash
wget https://dl.fbaipublicfiles.com/habitat/data/scene_datasets/gibson_habitat_trainval.zip
unzip -q gibson_habitat_trainval.zip
```
* You also need to download the [pointnav_gibson_v1](https://github.com/facebookresearch/habitat-lab/blob/main/DATASETS.md) task dataset. A task dataset consists of many episodes. Each episode includes initial pose of the agent, scene id etc. 
```bash 
mkdir pointnav_gibson_v1
cd pointnav_gibson_v1
wget https://dl.fbaipublicfiles.com/habitat/data/datasets/pointnav/gibson/v1/pointnav_gibson_v1.zip
unzip -q pointnav_gibson_v1.zip
```

# Run

- To run training with the baseline method ([active neural slam](https://arxiv.org/abs/2004.05155))

```bash
python main.py --config ./configs/train_NSLAM.txt
```

- For evaluation, run

```bash
python main.py --config ./configs/eval_vis.txt
```

You can change what saved models to evaluate in `./configs/eval_vis.txt`.  If `print_images = 1` images (camera observation + map) will saved into `$dum_location/$exp_name`. A video will also be generated from the saved images.  

## Running a Specific Scene for Evaluation

To evaluate on a specific scene, you first need to generate a list of all available validation episodes. You can do this by running the following script:

```bash
python utils/get_episodes.py
```

This will create a `val_episodes.json` file in the root directory, which contains the `episode_id` and `scene_id` for each episode in the validation set.

Once you have the `episode_id` of the scene you want to evaluate, add it to your evaluation configuration file (e.g., `configs/eval_lena.txt` or `configs/eval_NSLAM.txt`). Make sure you also have `num_processes = 1` in the config file.

```
eval_scene_id = YOUR_EPISODE_ID
```

Then, run the evaluation using that config file:

```bash
python main.py --config ./configs/eval_lena.txt
```
Replace `YOUR_EPISODE_ID` with the desired `episode_id` from the `val_episodes.json` file.

# Code structure
- You can find global policy, path planner, and local policy starting from `main.py` .
- The most important simulation environment file is `env/habitat/exploration_env.py` which defines the `step()` and `reset()` functions for each process.
- Multiple processes/threads/scenes can run in parallel, and this is defined in `env/__init__.py`.
- I know it’s a bit confusing, but there are three configuration files: `configs/train_lena.txt` for `main.py`, `env/habitat/configs/gibson.yaml` for the habitat simulation, and `env/habitat/configs/mapping.yaml` for the neural implicit map.
