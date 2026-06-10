# ULS AIMI Group 8

This repository contains the nnU-Net v2 training code, SLURM launch scripts, logs, and validation summaries for our ULS23 lesion segmentation augmentation experiments.

## Research Question

For ULS lesion segmentation with a lightweight nnU-Net baseline, which contributes more to validation performance — training duration or spatial augmentation strength?

We compared the default nnU-Net setup against several custom augmentation policies:

- StrongAug with moderate rotation, elastic deformation, scaling, and intensity augmentation.
- Wider rotation up to +/- 180 degrees.
- Stronger elastic deformation.
- Wider independent-axis scaling.
- Stronger intensity augmentation.
- Experimental lesion morphology augmentation, which failed during training and was excluded from the final results.

## Repository Structure

Each `run*` directory is one experiment. The folders are kept as experiment records: each contains the trainer code used for that run, the SLURM script, logs, validation output, and a short run-specific README.

| Directory | Experiment | Mean validation Dice |
| --- | --- | ---: |
| `run0_default_100ep/` | Default nnU-Net baseline, 100 epochs | 0.6102 |
| `run1_strongaug_100ep/` | StrongAug baseline, 100 epochs | 0.5199 |
| `run2_strongaug_300ep/` | StrongAug baseline, 300 epochs | **0.6269** |
| `run3_strongaug_v2_300ep_rot180/` | StrongAug with +/- 180 degree rotation, 300 epochs | 0.6234 |
| `run4_default_300ep/` | Default nnU-Net baseline, 300 epochs | 0.6209 |
| `run5_scaling_darko_100ep/` | StrongAug with wider independent-axis scaling, 100 epochs | 0.5032 |
| `run6_elastic_nuno_100ep/` | StrongAug with stronger elastic deformation, 100 epochs | 0.5046 |
| `run7_intensity_batoul_100ep/` | StrongAug with stronger intensity augmentation, 100 epochs | 0.5149 |
| `run8_morph_aksha_FAILED/` | Random lesion morphology augmentation, failed before epoch 0 | N/A |

## Key Findings

The best-performing configuration was `run2_strongaug_300ep`, which reached a mean validation Dice of **0.6269**.

The strongest overall signal was that training duration mattered: the 300-epoch StrongAug run improved substantially over the 100-epoch StrongAug run. At 300 epochs, StrongAug was only slightly better than the default nnU-Net baseline, suggesting that stronger augmentation helped, but the improvement was modest.

The ablations did not improve over the main StrongAug setup:

- Full +/- 180 degree rotation slightly reduced performance compared with the +/- 45 degree StrongAug run.
- Wider independent-axis scaling reduced performance in the 100-epoch ablation.
- Stronger elastic deformation reduced performance in the 100-epoch ablation.
- Stronger intensity augmentation was close to the 100-epoch StrongAug baseline but still lower.
- The morphology experiment stalled before training and was dropped from the reported results.

## Files Inside Each Run

Most experiment folders contain:

| File | Purpose |
| --- | --- |
| `README.txt` or `README.md` | Short description of the run, trainer, owner, configuration, and result |
| `customTrainers*.py` or `nnUNetTrainer*.py` | Custom nnU-Net trainer used for the experiment |
| `train_*.sh` | SLURM script used to launch or resume training |
| `training_log_*.txt` | Training log output |
| `validation_summary.json` | nnU-Net validation metrics, including foreground mean Dice |
| `debug.json` | nnU-Net debug and configuration metadata |

## Reproducing a Run

These experiments were run with nnU-Net v2 on a SLURM-managed GPU cluster. The dataset and trained checkpoints are not stored in this repository; the scripts expect the ULS23 data to already be prepared in nnU-Net format.

Before launching a run, make sure that:

1. nnU-Net v2 is installed in the active Python environment.
2. The custom trainer file for the run is available on the Python path used by nnU-Net.
3. The standard nnU-Net environment variables point to the correct dataset locations:

```bash
export nnUNet_raw="/path/to/nnUNet_raw"
export nnUNet_preprocessed="/path/to/nnUNet_preprocessed"
export nnUNet_results="/path/to/nnUNet_results"
```

Example training command for the best run:

```bash
nnUNetv2_train 400 3d_fullres 0 -tr nnUNetTrainer_ULS_300_StrongAug_aksha
```

On the cluster, use the corresponding SLURM script inside the run folder, for example:

```bash
sbatch run2_strongaug_300ep/train_fold0_300ep.sh
```

Some 300-epoch runs were resumed across multiple jobs using `checkpoint_latest.pth`.

## Validation Metrics

The main reported metric is `foreground_mean.Dice` from each `validation_summary.json`. These JSON files also include per-case metrics such as false positives, false negatives, IoU, true positives, true negatives, and Dice.
