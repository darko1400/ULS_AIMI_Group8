#!/bin/bash -e
# =============================================================================
# train_intensity_batoul_100ep.sh
# Batoul's ablation: stronger intensity augmentations
#   GaussianNoise variance: (0, 0.1) -> (0, 0.15)
#   Gamma range: (0.7, 1.5) -> (0.5, 2.0)
#   LowResolution scale: (0.5, 1) -> (0.4, 1)
# 100 epochs, fold 0, Dataset400_FSUP_ULS
# =============================================================================
#
# BEFORE SUBMITTING: change --mail-user below to your science username.
# Output files go to /home/$USER/training_jobs/ — make sure that folder exists:
#   mkdir -p ~/training_jobs
# =============================================================================

#SBATCH --job-name=intensity_batoul
#SBATCH --partition=csedu
#SBATCH --account=cseduimc037
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=16G
#SBATCH --time=08:00:00
#SBATCH --output=intensity_batoul_%j.out
#SBATCH --error=intensity_batoul_%j.err
#SBATCH --mail-user=CHANGE_ME_TO_YOUR_USERNAME
#SBATCH --mail-type=END,FAIL

echo "=== Started on $(hostname) at $(date) ==="

source /vol/csedu-nobackup/course/IMC037_aimi/group08/aksha_venv/bin/activate

export nnUNet_raw="/vol/csedu-nobackup/course/IMC037_aimi/group08/aimi-project/data/raw"
export nnUNet_preprocessed="/vol/csedu-nobackup/course/IMC037_aimi/group08/aimi-project/data/preprocessed"
export nnUNet_results="/vol/csedu-nobackup/course/IMC037_aimi/group08/aksha_results/nnUNet_results"

which python
python --version
nvidia-smi

CHECKPOINT_DIR="$nnUNet_results/Dataset400_FSUP_ULS/nnUNetTrainer_ULS_100_Intensity_batoul__nnUNetPlans__3d_fullres/fold_0"
if [ -f "$CHECKPOINT_DIR/checkpoint_latest.pth" ]; then
    echo "=== Resuming from checkpoint ==="
    nnUNetv2_train 400 3d_fullres 0 -tr nnUNetTrainer_ULS_100_Intensity_batoul --c
else
    echo "=== Starting fresh: 100ep with stronger intensity augs (Batoul ablation) ==="
    nnUNetv2_train 400 3d_fullres 0 -tr nnUNetTrainer_ULS_100_Intensity_batoul
fi

echo "=== Done at $(date) ==="
