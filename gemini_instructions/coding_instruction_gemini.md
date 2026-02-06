# Optimize the code to run faster 
* First, you will run @main.py with the configuration file @configs/eval_speedTest.txt to have a benchmark of how fast our current code is running. Feel free to add lines to @main.py to track how much time each part of the code takes. 
* Add this benchmark time performance to @gemini_instructions/code_optimization/GEMINI.md. 
* Before we optimize our codes to run faster, here are some rules/information:
    * Do not modify @configs/eval_speedTest.txt and @env/habitat/configs/mapping.yaml.
    * The main way to accelerate our codes is to use `torch.compile` as much as possible. 
    * Few examples where `torch.compile` can be applied to: @model.py, @env/habitat/ramen_mapping.py.
* Now we can do the optimization loop:
    * Describe to me your plan for optimizing the codes.
    * After getting my permission, implement the plan. 
    * Run run @main.py with the configuration file @configs/eval_speedTest.txt to get the new time performance.
    * Document the attempts you have maded and the time performance to  @gemini_instructions/code_optimization/GEMINI.md.
                                      