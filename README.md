# README

# Install

- First create the conda environmnet. *lena* stands for “learn to perform neural active mapping”. `python=3.9` is a must because `habitat` does not work with newer version python

```bash
conda create -n python=3.9 cmake=3.14.0
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
* Build Habitat-sim with CUDA 
```bash 
python setup.py install --with-cuda
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

# Code structure
- You can find global policy, path planner, and local policy starting from `main.py` .
- The most important simulation environment file is `env/habitat/exploration_env.py` which defines the `step()` and `reset()` functions for each process.
- Multiple processes/threads/scenes can run in parallel, and this is defined in `env/__init__.py`.
- I know it’s a bit confusing, but there are three configuration files: `configs/train_lena.txt` for `main.py`, `env/habitat/configs/gibson.yaml` for the habitat simulation, and `env/habitat/configs/mapping.yaml` for the neural implicit map.