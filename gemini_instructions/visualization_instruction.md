* You need to mainly look at @main.py and @env/habitat/exploration_env.py
* In addition to `self.full_map`, also visualize `self.local_map` of @utils/mapping_handler.py.
* Concatenate the four channels of `self.full_map` and `self.local_map` along the width dimension so we can display them as one image. 
* Now the visualization should have four rows, the frist and the second row would have three columns, while the third and the fourth row only have one colum, but they should all have the same width 