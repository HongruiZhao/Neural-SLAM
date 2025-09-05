# README

# Install

- First create the conda environmnet. *lena* stands for “learn to perform neural active mapping”. `python=3.9` is a must because `habitat` does not work with newer version python

```bash
conda create -n lena python=3.9
conda active lena
```

- Then install habitat sim and lab:

```bash
pip install habitat-lab habitat-baselines
conda install habitat-sim withbullet -c conda-forge -c aihabitat
```

- Install pytorch and torch vision. If you are using RTX 50 series GPU, it would be

```bash
pip3 install --pre torch torchvision clear--index-url https://download.pytorch.org/whl/nightly/cu128
```

- Now install the rest of the python packages.

```bash
pip install -r requirements.txt
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
- Multiple processes/threads/scenes can run in parallel, and this is defined in `env/**init**.py`.
- I know it’s a bit confusing, but there are three configuration files: `configs/train_lena.txt` for `main.py`, `env/habitat/configs/gibson.yaml` for the habitat simulation, and `env/habitat/configs/mapping.yaml` for the neural implicit map.