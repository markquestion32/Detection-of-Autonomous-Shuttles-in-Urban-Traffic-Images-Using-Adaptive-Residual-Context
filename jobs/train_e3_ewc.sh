#!/bin/bash
#SBATCH --account=def-gabilode
#SBATCH --job-name=e3_lwf
#SBATCH --output=e3_lwf-%j.out
#SBATCH --error=e3_lwf-%j.err
#SBATCH --time=12:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gpus-per-node=h100:1
#SBATCH --mem=64G

# E3: Learning without Forgetting (LwF)
# Simpler continual learning baseline using knowledge distillation

echo "============================================"
echo "E3: Learning without Forgetting (LwF)"
echo "============================================"
date

module load python/3.11 scipy-stack opencv/4.12.0
source /home/mayounes/scratch/myenv/bin/activate

cd /home/mayounes/scratch/project-mtl3

python src/train_lwf_correct.py --epochs 150 --alpha 1.0 --temperature 2.0

echo "E3 Complete"
date
