The main thing I want to investigate in this repo is how to quantify how well a robot has explored an area, and use the quantification to train an agent using RL to actively map unknon scenes. 

The most accurate method of quantifying the mapping quality is to do the evaluation using groundtruth meshes. I have implemented this in @eval/. However, this method is slow, requires access to previloged groundtruth information, and cannot be computed fast enough to use as a reward durng the RL training. 


The other two methods I have implelemted are the completion ratio and the uncertainty learned through the neural implicit mapping. Now I want to implement an ensemble-based uncertainty to see if it is as accurate as the evaluation-based method. 

First tell me what do you think about this research idea, and provide me some recent (>2023) research papers that also explore ways to quantify the exploration/mapping quality using uncertainty or other methods. 

Then help me implement the ensemble-based uncertainty. Here are some instructions:
(1) You would like to look into @env/habitat/model/scene_rep.py, @env/habitat/ramen_mapping.py, and @env/habitat/model/encodings.py. 
(2) I want to enable the ensemble uncertainty through ['grid']['uncertainty'] flag in @env/habitat/configs/mapping.yaml. We also need another flag to control the size of the ensmble. 
(3) The ensemble should be a set of differently initialized hash grid running in parallel. They take the same input `xyz_sampled` and give their own outputs. You should implement this into `HashUncertainty` class in @env/habitat/model/encodings.py. '
(4) The predictive uncertainty can be expressed as the variance over the individual member prediction.
(4) Once we get the uncertainty from the ensmble, this uncertainty is for the predictions at given spatial coordinates. Thus, we can fill the uncertainty values into the uncertainty grid using these coordinates. We should probably take the average of all historic uncertainty values ever filled into the same location. But I am also open to different implemetation option. 

Let's go and implement it!