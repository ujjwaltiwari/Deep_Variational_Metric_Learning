#!/bin/bash
#SBATCH -A research
#SBATCH -p long
#SBATCH --gres=gpu:1
#SBATCH --mem-per-cpu=10240
#SBATCH -n 5
#SBATCH --time=4-00:00:00
#SBATCH --mail-type=END

module add cuda/10.0

python3 trainer.py --log_dir logs/naive_triplet_sum_num_first_phase_epochs_25/ --num_first_phase_epochs 25 --data_root ~/.data/CUB_200_2011/images
