#!/bin/bash
#SBATCH --account=def-nsaunier
#SBATCH --mem=32G
#SBATCH --time=2:20:00
#SBATCH --output=infer_twin-%j.out
#SBATCH --error=infer_twin-%j.err
#SBATCH --cpus-per-task=4
#SBATCH --gpus-per-node=h100:1

# Load modules to match training
module load python/3.11 scipy-stack opencv/4.12.0
source /home/mayounes/scratch/myenv/bin/activate

cd /home/mayounes/scratch/project-mtl3

echo "Running Inference Verification..."
python src/joint_inference.py data/valid/images/vlcsnap-2024-01-26-13h11m05s491_png.rf.b5dcfaa07e8464a7ae223beab54500d7.jpg --weights /scratch/mayounes/project-mtl3/runs/detect/twin_v3_RESIDUAL_yolo11x_fixed_v6/weights/best.pt
