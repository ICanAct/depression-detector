#!/bin/bash
#SBATCH --job-name=text_mining_project_transformer
#SBATCH -o model_transformers_log.out
#SBATCH --gpus=a100:1
#SBATCH --partition=medium
#SBATCH --time=150
#SBATCH --nodelist=xgph5

nvidia-smi
source activate experiments
python3 -u depression-detector/main_transformer_run.py >> model_transformers_log.txt

