#!/bin/bash
#SBATCH --account=def-gabilode
#SBATCH --job-name=train_twin
#SBATCH --output=train_twin-%j.out
#SBATCH --error=train_twin-%j.err
#SBATCH --time=12:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gpus-per-node=h100:1
#SBATCH --mem=64G

# Train Twin Model (E4)
# Our proposed architecture with frozen veteran head

echo "============================================"
echo "Training E4: Twin Architecture"
echo "============================================"
date

module load python/3.11 scipy-stack opencv/4.12.0
source /home/mayounes/scratch/myenv/bin/activate

cd /home/mayounes/scratch/project-mtl3

python src/train_twin.py

echo ""
echo "============================================"
echo "Twin Training Complete"
echo "============================================"
date
