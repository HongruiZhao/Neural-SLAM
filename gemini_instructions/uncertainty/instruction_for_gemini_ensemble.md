# Instructions for code implementation
* Idea: instead of calling `update_uncert_grid` every time when `query_color_sdf` is called in @env/habitat/model/scene_rep.py, only update the uncertainty grid when:
    * At the last iteration of `first_frame_mapping` in @env/habitat/ramen_mapping.py
    * At the last tieration of `global_BA` in @env/habitat/ramen_mapping.py.
* Intuitions: 
    * ensemble uncertainty is usally trained first and then used during the test time to provide uncertainty estimation.
    * Since we are performing active mapping and we need real-time information of uncertainty to guide the policy, we cannot wait for the whole mapping process to end.
    * But we should at least wait for each step of mapping to finish so that the ensemble can properly learn from the new observation images. 
* You will first review if my idea of changing where the uncertainty should be updated is valid. Provide some research paper for reference if applicable. 
* Once we agree upon it, go ahead and implement the code. 