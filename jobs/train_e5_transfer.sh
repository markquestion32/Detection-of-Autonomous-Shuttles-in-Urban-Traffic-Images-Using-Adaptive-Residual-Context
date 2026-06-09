#!/bin/bash
#SBATCH --account=def-gabilode
#SBATCH --job-name=e5_transfer
#SBATCH --output=e5_transfer-%j.out
#SBATCH --error=e5_transfer-%j.err
#SBATCH --time=12:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gpus-per-node=h100:1
#SBATCH --mem=64G

# E5: Transfer Learning Baseline
# Freeze backbone, train only detection head

echo "============================================"
echo "E5: Transfer Learning Baseline"
echo "============================================"
date

# Activate environment
module load python/3.11 scipy-stack opencv/4.12.0
source /home/mayounes/scratch/myenv/bin/activate

cd /home/mayounes/scratch/project-mtl3

python src/train_transfer_learning.py --epochs 150

echo "E5 Complete"
date
