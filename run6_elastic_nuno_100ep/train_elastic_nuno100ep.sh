#!/bin/bash -e
# =============================================================================
# train_elastic_nuno_100ep.sh
# Nuno's ablation: stronger elastic deformation
#   p_elastic_deform: 0.25 -> 0.4
#   elastic_deform_magnitude: (8, 24) -> (10, 40)
# 100 epochs, fold 0, Dataset400_FSUP_ULS
# =============================================================================
#
# BEFORE SUBMITTING: change --mail-user below to your science username.
# Output files go to /home/$USER/training_jobs/ — make sure that folder exists:
#   mkdir -p ~/training_jobs
# =============================================================================

#SBATCH --job-name=elastic_nuno
#SBATCH --partition=csedu
#SBATCH --account=cseduimc037
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=16G
#SBATCH --time=08:00:00
#SBATCH --output=elastic_nuno_%j.out
#SBATCH --error=elastic_nuno_%j.err
#SBATCH --mail-user=CHANGE_ME_TO_YOUR_USERNAME
#SBATCH --mail-type=END,FAIL

echo "=== Started on $(hostname) at $(date) ==="

# Activate Akshaya's shared venv (read-only is sufficient to source/activate)
source /vol/csedu-nobackup/course/IMC037_aimi/group08/aksha_venv/bin/activate

# nnUNet paths — shared preprocessed data, results in aksha_results
# (Different trainer class name means no collision with anyone else's outputs)
export nnUNet_raw="/vol/csedu-nobackup/course/IMC037_aimi/group08/aimi-project/data/raw"
export nnUNet_preprocessed="/vol/csedu-nobackup/course/IMC037_aimi/group08/aimi-project/data/preprocessed"
export nnUNet_results="/vol/csedu-nobackup/course/IMC037_aimi/group08/aksha_results/nnUNet_results"

which python
python --version
nvidia-smi

CHECKPOINT_DIR="$nnUNet_results/Dataset400_FSUP_ULS/nnUNetTrainer_ULS_100_Elastic_nuno__nnUNetPlans__3d_fullres/fold_0"
if [ -f "$CHECKPOINT_DIR/checkpoint_latest.pth" ]; then
    echo "=== Resuming from checkpoint ==="
    nnUNetv2_train 400 3d_fullres 0 -tr nnUNetTrainer_ULS_100_Elastic_nuno --c
else
    echo "=== Starting fresh: 100ep with stronger elastic deformation (Nuno ablation) ==="
    nnUNetv2_train 400 3d_fullres 0 -tr nnUNetTrainer_ULS_100_Elastic_nuno
fi

echo "=== Done at $(date) ==="
