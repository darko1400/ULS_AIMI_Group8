Run 5: Independent-Axis Scaling Ablation
Trainer class: nnUNetTrainer_ULS_100_Scaling_darko
Owner: darkok
Mean Validation Dice: 0.5032

Change from StrongAug 300ep config (Run 2):
- p_scaling: 0.25 -> 0.4
- scaling range: (0.65, 1.6) -> (0.5, 1.8)
- p_synchronize_scaling_across_axes: 1 -> 0 (independent per axis)
- num_epochs: 300 -> 100

Result: Wider/independent scaling slightly HURTS performance at 100 epochs
(-0.0167 vs 100ep StrongAug baseline of 0.5199).
