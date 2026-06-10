#!/bin/bash -e
#SBATCH --job-name=aksha_train_300
#SBATCH --partition=csedu
#SBATCH --account=cseduimc037
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=16G
#SBATCH --time=08:00:00
#SBATCH --output=/home/akshayakumar/training_jobs/train300_%j.out
#SBATCH --error=/home/akshayakumar/training_jobs/train300_%j.err
#SBATCH --mail-user=akshayakumar
#SBATCH --mail-type=END,FAIL

echo "=== Started on $(hostname) at $(date) ==="

source /vol/csedu-nobackup/course/IMC037_aimi/group08/aksha_venv/bin/activate

export nnUNet_raw="/vol/csedu-nobackup/course/IMC037_aimi/group08/aimi-project/data/raw"
export nnUNet_preprocessed="/vol/csedu-nobackup/course/IMC037_aimi/group08/aimi-project/data/preprocessed"
export nnUNet_results="/vol/csedu-nobackup/course/IMC037_aimi/group08/aksha_results/nnUNet_results"

which python
python --version
nvidia-smi

CHECKPOINT_DIR="$nnUNet_results/Dataset400_FSUP_ULS/nnUNetTrainer_ULS_300_StrongAug_aksha__nnUNetPlans__3d_fullres/fold_0"
if [ -f "$CHECKPOINT_DIR/checkpoint_latest.pth" ]; then
    echo "=== Resuming from checkpoint ==="
    nnUNetv2_train 400 3d_fullres 0 -tr nnUNetTrainer_ULS_300_StrongAug_aksha --c
else
    echo "=== Starting fresh 300-epoch training ==="
    nnUNetv2_train 400 3d_fullres 0 -tr nnUNetTrainer_ULS_300_StrongAug_aksha
fi

echo "=== Done at $(date) ==="
