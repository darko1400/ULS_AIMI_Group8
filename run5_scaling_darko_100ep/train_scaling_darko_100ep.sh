#!/bin/bash -e
# =============================================================================
# train_scaling_darko_100ep.sh
# Darko's ablation: independent-axis scaling (lesions can become elongated)
#   p_scaling: 0.25 -> 0.4
#   scaling range: (0.65, 1.6) -> (0.5, 1.8)
#   p_synchronize_scaling_across_axes: 1 -> 0 (independent per axis)
# 100 epochs, fold 0, Dataset400_FSUP_ULS
# =============================================================================
#
# BEFORE SUBMITTING: change --mail-user below to your science username.
# Output files go to /home/$USER/training_jobs/ — make sure that folder exists:
#   mkdir -p ~/training_jobs
# =============================================================================

#SBATCH --job-name=scaling_darko
#SBATCH --partition=csedu
#SBATCH --account=cseduimc037
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=16G
#SBATCH --time=08:00:00
#SBATCH --output=scaling_darko_%j.out
#SBATCH --error=scaling_darko_%j.err
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

CHECKPOINT_DIR="$nnUNet_results/Dataset400_FSUP_ULS/nnUNetTrainer_ULS_100_Scaling_darko__nnUNetPlans__3d_fullres/fold_0"
if [ -f "$CHECKPOINT_DIR/checkpoint_latest.pth" ]; then
    echo "=== Resuming from checkpoint ==="
    nnUNetv2_train 400 3d_fullres 0 -tr nnUNetTrainer_ULS_100_Scaling_darko --c
else
    echo "=== Starting fresh: 100ep with independent-axis scaling (Darko ablation) ==="
    nnUNetv2_train 400 3d_fullres 0 -tr nnUNetTrainer_ULS_100_Scaling_darko
fi

echo "=== Done at $(date) ==="
