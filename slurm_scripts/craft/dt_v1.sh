#!/bin/bash
#SBATCH --job-name=dt-v1
#SBATCH --output=logs/dt-v1.log
#SBATCH --qos=normal
#SBATCH --time=15:00:00
#SBATCH --partition=rtx6000
#SBATCH --gres=gpu:1
#SBATCH --mem=16G

export PYTHONPATH=${PWD}
python -u experiment.py --env craft-causal --dataset four --max_iters 20 -w True --causal_dim 6 --causal_version v1