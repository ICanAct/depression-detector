#!/bin/bash
#SBATCH --job-name=text_mining_project_transformer
#SBATCH -o model_log.out
#SBATCH --gpus=a100:1
#SBATCH --partition=medium
#SBATCH --time=150

source activate experiments
python3 -u depression-detector/main_transformer_run.py >> model_log.txt

