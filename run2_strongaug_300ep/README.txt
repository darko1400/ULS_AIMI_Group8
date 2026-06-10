Run 2: StrongAug 300 epochs (BEST OVERALL CONFIGURATION)
Trainer class: nnUNetTrainer_ULS_300_StrongAug_aksha
Owner: akshayakumar
Mean Validation Dice: 0.6269

Augmentation: nnUNet StrongAug recipe with rotation range +/- 45 degrees.
Training duration: 300 epochs (resumed across multiple 8-hour SLURM jobs
using checkpoint_latest.pth).

This is the BEST configuration we found across all experiments. The
+0.107 Dice improvement over Run 1 (same recipe, 100 epochs) shows that
training duration is the dominant factor for this dataset.
