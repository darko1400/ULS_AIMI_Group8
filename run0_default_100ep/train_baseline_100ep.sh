#!/bin/bash -e
#SBATCH --partition=csedu
#SBATCH --account=cseduimc037
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=16G
#SBATCH --time=10:00:00
#SBATCH --output=/vol/csedu-nobackup/course/IMC037_aimi/group08/aimi-project/logs/baseline100_%j.out
#SBATCH --error=/vol/csedu-nobackup/course/IMC037_aimi/group08/aimi-project/logs/baseline100_%j.err
#SBATCH --mail-user=nsilva
#SBATCH --mail-type=END,FAIL

source /vol/csedu-nobackup/course/IMC037_aimi/group08/nsilva_venv/bin/activate

export nnUNet_raw="/vol/csedu-nobackup/course/IMC037_aimi/group08/aimi-project/data/raw"
export nnUNet_preprocessed="/vol/csedu-nobackup/course/IMC037_aimi/group08/aimi-project/data/preprocessed"
export nnUNet_results="/vol/csedu-nobackup/course/IMC037_aimi/group08/nsilva/nnUNet_results"

mkdir -p "$nnUNet_results"

echo "=== Baseline 100 epochs started on $(hostname) at $(date) ==="
nnUNetv2_train 400 3d_fullres 0 -tr nnUNetTrainer_100epochs
echo "=== Done at $(date) ==="
