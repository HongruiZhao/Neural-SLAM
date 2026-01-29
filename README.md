# LUNA: Learning with Uncertainty to Perform Neural Active Mapping

<p align="center">
  <img src="https://brand.illinois.edu/wp-content/uploads/2024/02/Color-Variation-Orange-Block-I-Blue-Background-1.png" width="75" />
  &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;
  <img src="https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcT8jIgqKkbqA8jL_rzz8-x-MM05TtjfmVFuLQ&s" width="75" />
    &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;
  <img src="https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcQvXarWL-vU2OWNMtGfRYyP7L0C-GLTePn4bw&s" width="75" />
</p>



## Table of Contents
- [Install](#install)
    - [Install habitat sim from source](#install-habitat-sim-from-source)
    - [Install habitat lab and habitat baseline](#install-habitat-lab-and-habitat-baseline)
    - [Install packages for lena](#install-packages-for-lena)
- [Download Gibson dataset](#download-gibson-dataset)
- [Run](#run)
    - [Run for training and evaluation](#run-for-training-and-evaluation)
    - [Running a Specific Scene for Evaluation](#running-a-specific-scene-for-evaluation)
- [Ensemble Uncertainty](#ensemble-uncertainty)
- [Automated Evaluation and Plotting](#automated-evaluation-and-plotting)
- [Code structure](#code-structure)

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

## Run for training and evaluation
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


# Ensemble Uncertainty

This project implements predictive uncertainty estimation using **Deep Ensembles**. The uncertainty is estimated using the empirical variance (epistemic uncertainty) of predictions from multiple independently trained neural networks. This captures disagreement among ensemble members, serving as a proxy for the model's uncertainty.

For a detailed explanation of the theory, implementation details (such as training vs. inference behavior), and key files involved, please refer to [gemini_instructions/uncertainty/GEMINI.md](gemini_instructions/uncertainty/GEMINI.md).

# Automated Evaluation and Plotting

* To automatically evaluate the reconstruction quality (accuracy, completion, and completion ratio) for a series of meshes, run the script below. 
    * This script will cull occluded parts of the meshes (unless `--skip_cull` is used), compare them against the ground truth, and save the metrics to `eval/evaluation_results.json`.
    * To have meshed saved when running `main.py`, change `[mesh][vis]` in `env/habitat/configs/mapping.yaml`. `[mesh][vis]=10` means saving a mesh every 10 iterations. 

    ```bash
    python eval/auto_eval.py --config eval/eval_basic.yaml
    ```

* To plot the evaluation results against uncertainty and scene coverage, run the code below.
    * Replace `<experiment_name>` with the experiment name (folder name inside `results/dump/`). This generates `evaluation_combined.jpg`.
    ```bash
    cd eval
    python plot_eval_results.py --eval evaluation_results --exp <experiment_name>
    ```


# Code structure
- You can find global policy, path planner, and local policy starting from `main.py` .
- The most important simulation environment file is `env/habitat/exploration_env.py` which defines the `step()` and `reset()` functions for each process.
- Multiple processes/threads/scenes can run in parallel, and this is defined in `env/__init__.py`.
- I know it’s a bit confusing, but there are three configuration files: `configs/train_lena.txt` for `main.py`, `env/habitat/configs/gibson.yaml` for the habitat simulation, and `env/habitat/configs/mapping.yaml` for the neural implicit map.
