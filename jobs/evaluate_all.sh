#!/bin/bash
#SBATCH --account=def-gabilode
#SBATCH --job-name=evaluate_all
#SBATCH --output=evaluate_all-%j.out
#SBATCH --error=evaluate_all-%j.err
#SBATCH --time=2:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gpus-per-node=h100:1
#SBATCH --mem=32G

# Evaluate All Trained Models
# Includes E0 (Pretrained YOLO baseline)

echo "============================================"
echo "Evaluating All Models"
echo "============================================"
date

# Activate environment
module load python/3.11 scipy-stack opencv/4.12.0
source /home/mayounes/scratch/myenv/bin/activate

cd /home/mayounes/scratch/project-mtl3

# Create results directory
mkdir -p results

SHUTTLE_TEST="data/test/images"
COCO_VAL="data/val2017"

# E0: Pretrained YOLO11x (No training - baseline on COCO)
echo ""
echo ">>> Evaluating E0: Pretrained YOLO11x (Baseline) <<<"
python src/evaluate_experiment.py \
    --weights yolo11x.pt \
    --model-type pretrained \
    --shuttle-path $SHUTTLE_TEST \
    --coco-path $COCO_VAL \
    --name e0_pretrained \
    --output results

# E1: Upper Bound (Joint Training)
echo ""
echo ">>> Evaluating E1: Upper Bound (Joint Training) <<<"
python src/evaluate_experiment.py \
    --weights runs/detect/e1_upperbound/weights/best.pt \
    --model-type finetuned \
    --shuttle-path $SHUTTLE_TEST \
    --coco-path $COCO_VAL \
    --name e1_upper \
    --output results

# E2: Naive fine-tuning (81 classes)
echo ""
echo ">>> Evaluating E2: Naive Fine-tuning (81 classes) <<<"
python src/evaluate_experiment.py \
    --weights runs/detect/e2_naive/weights/best.pt \
    --model-type finetuned \
    --shuttle-path $SHUTTLE_TEST \
    --coco-path $COCO_VAL \
    --name e2_naive \
    --output results

# E3: LwF (81 classes)
echo ""
echo ">>> Evaluating E3: Learning without Forgetting (81 classes) <<<"
python src/evaluate_experiment.py \
    --weights runs/detect/e3_lwf/weights/best.pt \
    --model-type finetuned \
    --shuttle-path $SHUTTLE_TEST \
    --coco-path $COCO_VAL \
    --name e3_lwf \
    --output results

# E4: Twin Architecture
echo ""
echo ">>> Evaluating E4: Twin Architecture (Ours) <<<"
python src/evaluate_experiment.py \
    --weights runs/detect/twin/weights/best.pt \
    --model-type twin \
    --shuttle-path $SHUTTLE_TEST \
    --coco-path $COCO_VAL \
    --name e4_twin \
    --output results

# E5: Transfer Learning (Frozen Backbone)
echo ""
echo ">>> Evaluating E5: Transfer Learning (Frozen Backbone) <<<"
python src/evaluate_experiment.py \
    --weights runs/detect/e5_transfer/weights/best.pt \
    --model-type finetuned \
    --shuttle-path $SHUTTLE_TEST \
    --coco-path $COCO_VAL \
    --name e5_transfer \
    --output results

# E6: Pure Shuttle Detector (1 class output)
echo ""
echo ">>> Evaluating E6: Pure Shuttle Detector <<<"
python src/evaluate_experiment.py \
    --weights runs/detect/e6_shuttle_only/weights/best.pt \
    --model-type finetuned \
    --shuttle-path $SHUTTLE_TEST \
    --coco-path $COCO_VAL \
    --name e6_shuttle \
    --output results

# Generate comparison
echo ""
echo ">>> Generating Results Comparison <<<"
python src/generate_results.py --results-dir results

echo ""
echo "============================================"
echo "Evaluation Complete"
echo "============================================"
date
