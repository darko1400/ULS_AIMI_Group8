Run 3: StrongAug 300 epochs with wider rotation
Trainer class: nnUNetTrainer_ULS_300_StrongAug_v2_aksha
Owner: akshayakumar
Mean Validation Dice: 0.6234

Augmentation: same as Run 2 except rotation range widened from
+/- 45 degrees to +/- 180 degrees (full rotation).
Training duration: 300 epochs (multiple resumed SLURM jobs).

Result: wider rotation slightly HURTS performance (-0.0035 vs Run 2).
This suggests moderate rotation (+/- 45 degrees) is closer to optimal
for the ULS23 dataset; the model does not benefit from learning to
handle arbitrary rotations on this task.
