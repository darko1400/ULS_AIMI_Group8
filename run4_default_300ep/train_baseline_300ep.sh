#!/bin/bash -e
#SBATCH --job-name=aksha_base_300
#SBATCH --partition=csedu
#SBATCH --account=cseduimc037
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=16G
#SBATCH --time=08:00:00
#SBATCH --output=/home/akshayakumar/training_jobs/base300_%j.out
#SBATCH --error=/home/akshayakumar/training_jobs/base300_%j.err
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

CHECKPOINT_DIR="$nnUNet_results/Dataset400_FSUP_ULS/nnUNetTrainer_default_300_aksha__nnUNetPlans__3d_fullres/fold_0"
if [ -f "$CHECKPOINT_DIR/checkpoint_latest.pth" ]; then
    echo "=== Resuming from checkpoint ==="
    nnUNetv2_train 400 3d_fullres 0 -tr nnUNetTrainer_default_300_aksha --c
else
    echo "=== Starting fresh baseline 300-epoch training (default nnUNet, no StrongAug) ==="
    nnUNetv2_train 400 3d_fullres 0 -tr nnUNetTrainer_default_300_aksha
fi

echo "=== Done at $(date) ==="
