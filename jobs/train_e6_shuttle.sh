#!/bin/bash
#SBATCH --account=def-gabilode
#SBATCH --job-name=e6_shuttle
#SBATCH --output=e6_shuttle-%j.out
#SBATCH --error=e6_shuttle-%j.err
#SBATCH --time=12:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gpus-per-node=h100:1
#SBATCH --mem=64G

# E6: Pure Shuttle Detector
# Simple baseline - YOLO trained only on shuttles

echo "============================================"
echo "E6: Pure Shuttle Detector"
echo "============================================"
date

module load python/3.11 scipy-stack opencv/4.12.0
source /home/mayounes/scratch/myenv/bin/activate

cd /home/mayounes/scratch/project-mtl3

python src/train_shuttle_only.py --epochs 150

echo "E6 Complete"
date
