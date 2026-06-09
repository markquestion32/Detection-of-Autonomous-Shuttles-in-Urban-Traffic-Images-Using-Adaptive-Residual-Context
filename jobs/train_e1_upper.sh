#!/bin/bash
#SBATCH --account=def-gabilode
#SBATCH --job-name=e1_upper
#SBATCH --output=e1_upper-%j.out
#SBATCH --error=e1_upper-%j.err
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gpus-per-node=h100:1
#SBATCH --mem=64G

# E1: Upper Bound (COCO + Shuttle Joint Training)
# Requires data/data_combined/ with COCO + shuttle images

echo "============================================"
echo "E1: Upper Bound (COCO + Shuttle)"
echo "============================================"
date

module load python/3.11 scipy-stack opencv/4.12.0
source /home/mayounes/scratch/myenv/bin/activate

cd /home/mayounes/scratch/project-mtl3

python src/train_e1_upperbound.py --epochs 50

echo "E1 Complete"
date
