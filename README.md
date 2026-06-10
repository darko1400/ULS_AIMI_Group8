# ULS_AIMI_Group8

This repository contains the training scripts, logs, validation summaries, and supporting files for experiments on the ULS23 course baseline model.

## Research Question

Does the ULS course baseline overfit to the limited training data, and can stronger spatial augmentation reduce this overfitting and improve generalization to unseen lesion shapes?

The experiments compare the default nnU-Net augmentation policy against stronger spatial augmentation settings, including wider rotation, stronger elastic deformation, and wider scaling.

## Repository Structure

Each `run*` directory corresponds to one experiment.

| Directory | Description |
| --- | --- |
| `run0_default_100ep/` | Default nnU-Net baseline trained for 100 epochs |
| `run1_strongaug_100ep/` | StrongAug trainer trained for 100 epochs |
| `run2_strongaug_300ep/` | StrongAug trainer trained for 300 epochs |
| `run3_strongaug_v2_300ep_rot180/` | StrongAug trainer with wider ±180° rotation trained for 300 epochs |
| `run4_default_300ep/` | Default nnU-Net baseline trained for 300 epochs |
| `run5_scaling_darko_100ep/` | StrongAug variant with wider/independent scaling, trained for 100 epochs |
| `run6_elastic_nuno_100ep/` | StrongAug variant with stronger elastic deformation, trained for 100 epochs |
| `run7_intensity_batoul_100ep/` | Intensity augmentation experiment trained for 100 epochs |
| `run8_morph_aksha_FAILED/` | Failed morphology augmentation experiment |

## Files Inside Each Run Directory

Most run folders contain:

| File | Purpose |
| --- | --- |
| `README.txt` | Short description of the run configuration and validation result |
| `train_*.sh` | Script used to launch training on the cluster |
| `training_log_*.txt` | Training output logs |
| `validation_summary.json` | Validation metrics, including mean Dice |
| `debug.json` | Debug/configuration information |
| `progress.png` | Training progress plot, if available |
