#!/bin/bash -e
#SBATCH --job-name=aksha_train_f0
#SBATCH --partition=csedu
#SBATCH --account=cseduimc037
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=16G
#SBATCH --time=08:00:00
#SBATCH --output=/home/akshayakumar/training_jobs/train_%j.out
#SBATCH --error=/home/akshayakumar/training_jobs/train_%j.err
#SBATCH --mail-user=akshayakumar
#SBATCH --mail-type=END,FAIL

echo "=== Started on $(hostname) at $(date) ==="

# Activate YOUR venv
source /vol/csedu-nobackup/course/IMC037_aimi/group08/aksha_venv/bin/activate

# nnUNet environment variables (point to existing preprocessed data)
export nnUNet_raw="/vol/csedu-nobackup/course/IMC037_aimi/group08/aimi-project/data/raw"
export nnUNet_preprocessed="/vol/csedu-nobackup/course/IMC037_aimi/group08/aimi-project/data/preprocessed"
export nnUNet_results="/vol/csedu-nobackup/course/IMC037_aimi/group08/aksha_results/nnUNet_results"

# Sanity checks
which python
python --version
echo "nnUNet_raw=$nnUNet_raw"
echo "nnUNet_preprocessed=$nnUNet_preprocessed"
echo "nnUNet_results=$nnUNet_results"
nvidia-smi

# Resume from checkpoint if it exists, otherwise start fresh
CHECKPOINT_DIR="$nnUNet_results/Dataset400_FSUP_ULS/nnUNetTrainer_ULS_100_StrongAug_aksha__nnUNetPlans__3d_fullres/fold_0"
if [ -f "$CHECKPOINT_DIR/checkpoint_latest.pth" ]; then
    echo "=== Resuming from checkpoint ==="
    nnUNetv2_train 400 3d_fullres 0 -tr nnUNetTrainer_ULS_100_StrongAug_aksha --c
else
    echo "=== Starting fresh training ==="
    nnUNetv2_train 400 3d_fullres 0 -tr nnUNetTrainer_ULS_100_StrongAug_aksha
fi

echo "=== Done at $(date) ==="
