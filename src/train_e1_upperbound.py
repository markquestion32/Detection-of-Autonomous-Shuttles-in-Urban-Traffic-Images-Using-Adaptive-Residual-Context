"""
train_e1_upperbound.py - E1: Upper Bound (COCO + Shuttle Joint Training)

Trains YOLO on COCO training data + shuttle data together.
This represents the best possible performance when you have access to all data.

NOTE: Requires COCO train2017 images in data/coco_train2017/
"""

import os
import sys
from ultralytics import YOLO

# Path setup
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)

# Use combined COCO + Shuttle config
DATA_PATH = os.path.join(project_root, "data/data_combined.yaml")


def train_upperbound(epochs=50):
    """
    Train on combined COCO + Shuttle dataset.
    
    This is the "upper bound" - what performance looks like when you
    have access to both the original task (COCO) and new task (shuttle)
    during training.
    """
    print(">>> E1: Upper Bound (COCO + Shuttle Joint Training) <<<")
    print(f"Data: {DATA_PATH}")
    
    # Check if combined dataset exists
    if not os.path.exists(DATA_PATH):
        print("ERROR: data_combined.yaml not found!")
        print("You need to set up the combined COCO + Shuttle dataset first.")
        return
    
    # Start from pretrained COCO weights
    model = YOLO("yolo11x.pt")
    
    # Train on combined data
    model.train(
        data=DATA_PATH,
        epochs=epochs,
        imgsz=640,
        batch=16,
        project=os.path.join(project_root, "runs/detect"),
        name="e1_upperbound",
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
    
    print(">>> E1 Training Complete <<<")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=50)
    args = parser.parse_args()
    
    train_upperbound(epochs=args.epochs)
