I would like you to modify auto_eval.py to evaluate reconstrution quality of meshese generated during the mapping process in main.py.
I will first provide you some context about the meshes needed to be evaluated, then I will define the details of tasks I want you to di. 

How mesheses are generated and saved:
(1) In "env/habitat/configs/mapping.yaml", mesh-vis will determine how often a mesh is generated from the neural implicit mapping module. 
(2) In "env/habitat/ramen_mapping.py", function "save_mesh" will extract and save the mesh. The mesh save path is also defined here. 

What are the ground-truth meshes:
(1) All ground-truth meshes are saved in "/home/hongrui/Datasets/habitat/scene_datasets/gibson" in the format of scene_name + .glb. For example,
"Adrain.glb" for the scene Adrian. 
(2) the scene an agent is exploring at the current episode can be accquired by "self.scene_name" attribute of an "Exploration_Env" object (env/habitat/exploration_env.py).
