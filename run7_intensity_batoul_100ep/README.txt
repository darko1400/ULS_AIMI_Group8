Run 7: Stronger Intensity Augmentation Ablation
Trainer class: nnUNetTrainer_ULS_100_Intensity_batoul
Owner: batoulh
Mean Validation Dice: 0.5148970284443632

Change from StrongAug 300ep config (Run 2):
- GaussianNoise variance: (0, 0.1) -> (0, 0.15)
- Gamma range: (0.7, 1.5) -> (0.5, 2.0) [applied to BOTH Gamma transforms]
- SimulateLowResolution scale: (0.5, 1) -> (0.4, 1)
- num_epochs: 300 -> 100
