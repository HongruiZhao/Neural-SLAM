# Evaluation Improvements (February 2026)

## Metrics Shift
We have transitioned from basic 3D distance metrics (Accuracy, Completion) to more robust Point-to-Point metrics using a 5cm threshold, which better reflect the quality of the reconstructed surface for navigation tasks.

- **Precision 5cm**: The percentage of reconstructed surface points whose nearest ground-truth point lies within a 5 cm threshold. This measures the "correctness" of the reconstruction.
- **Recall 5cm**: The percentage of ground-truth points whose nearest reconstructed point lies within a 5 cm threshold. This measures the "completeness" of the reconstruction.
- **F1 5cm**: The harmonic mean of Precision and Recall, providing a single balanced score for reconstruction quality.

## Spatially-Resolved 2D Evaluation
To better understand where the model succeeds or fails, we now project these 3D metrics onto a 2D top-down grid.

- **Grid Mapping**: Points are discretized into a 2D grid matching the resolution and bounds of the uncertainty map.
- **Visualizations**: For every evaluated mesh, a subplot is generated containing:
    1. **Precision Map**: Heatmap of precision across the scene.
    2. **Recall Map**: Heatmap of recall/coverage.
    3. **F1 Map**: Combined quality metric.
- **Storage**: Results are saved in `eval_vis_results/{exp_name}/{mesh_name}.png`.

## Automated Pipeline and Utilities
- **`eval/auto_eval.py`**: Updated to automatically parse and store the new Precision/Recall/F1 metrics into JSON result files.
- **`eval/make_video.py`**: A new utility to compile the spatial metric plots into a video, allowing for visualization of how reconstruction quality improves over time.
- **Improved Alignment**: The `get_align_transformation` in `eval_recon.py` has been refactored to use the agent's initial state for more reliable alignment between Habitat coordinates and the extracted mesh.
