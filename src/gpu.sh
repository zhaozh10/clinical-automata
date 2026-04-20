#!/bin/bash
#SBATCH --partition=radcluster
#SBATCH --job-name=auto
#SBATCH --time=5-00:00:00
#SBATCH --nodelist=hawking
#SBATCH --cpus-per-task=6
#SBATCH --gres=gpu:4
#SBATCH --mem=120G

nvidia-smi
sleep infinity
# source ~/llama/bin/activate


