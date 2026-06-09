"""
train_naive_81class.py - E2: Naive Fine-tuning with 81 classes

Fine-tunes YOLO11x on shuttle data while keeping all 81 class outputs.
This properly demonstrates catastrophic forgetting (weights drift, not architecture change).
"""

import os
import sys
from ultralytics import YOLO

# Path setup
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)

# Use 81-class config (shuttle = class 80)
DATA_PATH = os.path.join(project_root, "data/data_81class.yaml")


def train_naive_81class(model_name="yolo11x.pt", epochs=50):
    """
    Naive fine-tuning on shuttle dataset with 81 classes.
    
    The model keeps all 81 COCO class outputs, but training only provides
    shuttle labels. This causes catastrophic forgetting of COCO classes
    through weight drift (not architecture change).
    """
    print(f">>> E2: Naive Fine-tuning (81 classes) <<<")
    print(f"Model: {model_name}")
    print(f"Data: {DATA_PATH}")
    print("NOTE: Shuttle = class 80, COCO = classes 0-79")
    
    # Load pretrained COCO model
    model = YOLO(model_name)
    
    # Fine-tune on shuttle data
    model.train(
        data=DATA_PATH,
        epochs=epochs,
        imgsz=640,
        batch=16,
        project=os.path.join(project_root, "runs/detect"),
        name="e2_naive",
        exist_ok=True,
        device=0,
        workers=8,
        plots=True,
        save=True,
        val=True,
        optimizer='AdamW',
        lr0=0.001,
        patience=25
    )
    
    print(">>> E2 Training Complete <<<")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=50)
    args = parser.parse_args()
    
    train_naive_81class(epochs=args.epochs)
