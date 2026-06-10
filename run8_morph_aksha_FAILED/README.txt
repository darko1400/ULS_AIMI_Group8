Run 8: Random Lesion Morphology Ablation (FAILED — DROPPED FROM RESULTS)

Trainer class: nnUNetTrainer_ULS_100_Morph_aksha
Owner: akshayakumar
Status: FAILED — never reached epoch 0
Result: N/A

Description:
Custom augmentation that randomly dilates or erodes the lesion
segmentation mask using scipy's binary_dilation / binary_erosion (radius
1-3 voxels, p=0.3 per sample). The intent was to inject label-noise as
a regularizer to improve generalization to unseen lesion boundaries.

What happened:
The job (SLURM ID 10298848) ran for 1h 48min wall time without
producing any epoch-level output. Initialization completed normally
(data loaders set up, splits loaded, GPU detected) but training stalled
before Epoch 0. No errors were raised in stderr. The job was eventually
cancelled.

Suspected root cause:
The scipy-based morphological operations introduced contention or
deadlock within nnU-Net's multi-process data loader pipeline. Each
worker process must apply this transform on the CPU before passing
batches to the GPU. With multiple workers attempting slow per-batch
scipy operations on 3D arrays simultaneously, the data pipeline stalled.

Files in this folder:
- morph_aksha_10298848.out: stdout from the failed job (initialization successful, stalled at "using pin_memory on device 0")
- morph_aksha_10298848.err: empty (no Python errors raised)
- customTrainersULS_morph_aksha.py: the custom trainer class
- random_lesion_morphology.py: the morphology transform (with __init__ fix and torch<->numpy conversion)
- train_morph_aksha_100ep.sh: the SLURM submission script

This ablation is omitted from the reported results and documented as a
limitation. Possible remediation strategies for future work:
1. Reimplement morphology in PyTorch (e.g., F.max_pool3d for dilation,
   F.avg_pool3d-based for erosion) to avoid scipy + cross-process tensor
   conversion overhead.
2. Lower p_per_sample to reduce frequency of slow scipy ops.
3. Instrument the data loader pipeline to pinpoint the bottleneck.
