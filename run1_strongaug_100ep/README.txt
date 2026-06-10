Run 1: StrongAug 100 epochs baseline
Trainer class: nnUNetTrainer_ULS_100_StrongAug_aksha
Owner: akshayakumar
Mean Validation Dice: 0.5199

Augmentation: nnUNet StrongAug recipe with rotation range +/- 45 degrees.
Training duration: 100 epochs (one 8-hour SLURM job).

This is the 100-epoch baseline used as the reference point for all
single-change ablations (Runs 5, 6, 7). All ablations branch from this
config with one parameter changed.
