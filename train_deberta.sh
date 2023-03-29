#!/bin/bash
#SBATCH --job-name=text_mining_project_deberta
#SBATCH -o deberta_log.out
#SBATCH --gpus=a100:1
#SBATCH --partition=medium
#SBATCH --time=150
#SBATCH --nodelist=xpgh5

nvidia-smi
source activate experiments
python3 -u depression-detector/main_deberta_run.py >> output_deberta.txt

