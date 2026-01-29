# Instructions for code implementation
* Extend our current ensemble implementation so that, instead of using a single decoder, creates a sepearte decoder for each memeber of the ensemble to ensure the full independence. 
* Add a flag to switch between a single shared decoder and multiple decoders. Aftet the implemetation.
* Add this flag to @./env/habitat/configs/mapping.yaml under `grid`.
* Update @gemini_instructions/uncertainty/GEMINI.md file to reflect the new implementation.

# Instructions for running and testing the implementation 
* After finish the implementation, modify @./configs/eval_NSLAM.txt to give it a proper `exp_name`.
* Run `python main.py --config ./configs/eval_NSLAM.txt`.
* If any mistake arises, debug until the code can be excuted succesfully from start to finish. 