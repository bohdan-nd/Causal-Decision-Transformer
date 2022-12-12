#!/bin/bash
#SBATCH --job-name=original-dt
#SBATCH --output=logs/dt-causal.log
#SBATCH --qos=normal
#SBATCH --time=15:00:00
#SBATCH --partition=rtx6000
#SBATCH --gres=gpu:1
#SBATCH --mem=16G

export PYTHONPATH=${PWD}
python -u experiment.py --env craft --dataset four --max_iters 20 -w True