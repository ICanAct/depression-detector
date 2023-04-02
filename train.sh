#!/bin/bash
#SBATCH --job-name=text_mining_project_transformer
#SBATCH -o model_deberta_log.out
#SBATCH --gpus=a100:1
#SBATCH --partition=medium
#SBATCH --time=150
#SBATCH --nodelist=xgph8

nvidia-smi
source activate experiments
python3 -u depression-detector/main_deberta_run.py >> model_deberta_log.txt

