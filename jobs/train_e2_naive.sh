#!/bin/bash
#SBATCH --account=def-gabilode
#SBATCH --job-name=e2_naive
#SBATCH --output=e2_naive-%j.out
#SBATCH --error=e2_naive-%j.err
#SBATCH --time=12:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gpus-per-node=h100:1
#SBATCH --mem=64G

# E2: Naive Fine-tuning (81 classes)
# Shows catastrophic forgetting

echo "============================================"
echo "E2: Naive Fine-tuning (81 classes)"
echo "============================================"
date

module load python/3.11 scipy-stack opencv/4.12.0
source /home/mayounes/scratch/myenv/bin/activate

cd /home/mayounes/scratch/project-mtl3

python src/train_naive_81class.py --epochs 150

echo "E2 Complete"
date
