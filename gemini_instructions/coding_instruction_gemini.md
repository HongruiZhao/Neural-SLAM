# Instructions for code implementation

## Improve and debug the code 
* Now my ensemble uncertainty, detailed in @gemini_instructions/uncertainty/GEMINI.md, has been implemented correctly, I will start doing RL training with this uncertainty to achieve better scene exploration.
* My training will use the configuration file @configs/train_lena.txt.
* You will understand how the RL training is handled currently and add a summary to @gemini_instructions/training/GEMINI.md. 
* You will first verify my RL training workflow using uncertainty map and uncertainty rewards.
    * Identify and fix all bugs you can find.
    * Based on your knowledge of reinforcement learning for navigation/active mapping, propose improvements to the current RL training workflow such as better reward calculation. Provide relevant papers to back up your proposed improvements.
* You will help me accelerate my implementation:
    * Write a test code to run one episode of training, and then profile it with cProfile.
    * Analyze the profile and identify the parts of the code that cost long time to compute.
    * Try to accelerate the code. For example, is habitat vectorized environment is implemented correctly to ensure fast vectorized training?
* Only start implementing your proposed fixes/improvements after I have reviewed and agreeed with what you proposed. 
* Finally, update @gemini_instructions/training/GEMINI.md with the all implemented improvements/fixes/



                                      