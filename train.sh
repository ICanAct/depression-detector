#!/bin/bash
#SBATCH --job-name=text_mining_project
#SBATCH -o model_log.out
#SBATCH --gpus=a100:1
#SBATCH --partition=medium
#SBATCH --time=4320

conda activate experiments
python3 -u .trainers/transformers_trainer.py >> model_log.txt

