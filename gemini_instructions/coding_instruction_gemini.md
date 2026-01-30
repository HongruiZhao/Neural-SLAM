# Instructions for code implementation
* Write a script that can automatically run @main.py multiple times to conduct a series of experiments.
* Each experiment would have different global configuration files (stored in @configs) and different mapping configuration file @env/habitat/configs/mapping.yaml.
* The script should allow me to choose what configuration files to use for each experiment, and also allow me to change the parameters inside the configuration files.
    * An example: I choose to use @configs/eval_NSLAM.txt for all three experiments. There is the `exp_name` in the global config file. I would provide a list of names `['exp_1' ,'exp_2', 'exp_3']`, and the `exp_name` would take value from this list sequentially for each experiment. 
* Now @main.py will save a 'train.log' file under the specific experiment folder in @results/models. Instead, I simply want to just save the global and the mapping configuration files into the folder.
