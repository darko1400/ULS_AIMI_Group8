#!/bin/bash -e
# =============================================================================
# train_morph_aksha_100ep.sh
# Akshaya's ablation: + RandomLesionMorphologyTransform
# 100 epochs, fold 0, Dataset400_FSUP_ULS
# =============================================================================

#SBATCH --job-name=morph_aksha
#SBATCH --partition=csedu
#SBATCH --account=cseduimc037
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=16G
#SBATCH --time=08:00:00
#SBATCH --output=/home/akshayakumar/training_jobs/morph_aksha_%j.out
#SBATCH --error=/home/akshayakumar/training_jobs/morph_aksha_%j.err
#SBATCH --mail-user=akshayakumar
#SBATCH --mail-type=END,FAIL

echo "=== Started on $(hostname) at $(date) ==="

# Activate Akshaya's shared venv (read access is sufficient to activate)
source /vol/csedu-nobackup/course/IMC037_aimi/group08/aksha_venv/bin/activate

# nnUNet paths — shared preprocessed data, results in aksha_results
export nnUNet_raw="/vol/csedu-nobackup/course/IMC037_aimi/group08/aimi-project/data/raw"
export nnUNet_preprocessed="/vol/csedu-nobackup/course/IMC037_aimi/group08/aimi-project/data/preprocessed"
export nnUNet_results="/vol/csedu-nobackup/course/IMC037_aimi/group08/aksha_results/nnUNet_results"

which python
python --version
nvidia-smi

CHECKPOINT_DIR="$nnUNet_results/Dataset400_FSUP_ULS/nnUNetTrainer_ULS_100_Morph_aksha__nnUNetPlans__3d_fullres/fold_0"
if [ -f "$CHECKPOINT_DIR/checkpoint_latest.pth" ]; then
    echo "=== Resuming from checkpoint ==="
    nnUNetv2_train 400 3d_fullres 0 -tr nnUNetTrainer_ULS_100_Morph_aksha --c
else
    echo "=== Starting fresh: 100ep with RandomLesionMorphologyTransform (Akshaya ablation) ==="
    nnUNetv2_train 400 3d_fullres 0 -tr nnUNetTrainer_ULS_100_Morph_aksha
fi

echo "=== Done at $(date) ==="
