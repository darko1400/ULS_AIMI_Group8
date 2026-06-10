Run 6: Stronger Elastic Deformation Ablation
Trainer class: nnUNetTrainer_ULS_100_StrongElastic_nuno
Owner: nsilva
Mean Validation Dice: 0.5045544907711373

Change from StrongAug 300ep config (Run 2):
- p_elastic_deform: 0.2 -> 0.4
- elastic_deform_magnitude: (8.0, 24.0) -> (10.0, 40.0)
- num_epochs: 300 -> 100

(Note: source path is in /nsilva/ not /aksha_results/ because Nuno ran it
from his own user folder.)
