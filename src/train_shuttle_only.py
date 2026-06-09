"""
E6: Pure Shuttle Detector (Baseline)
Train YOLO11x on shuttle data ONLY, with single-class output.

This is the simplest possible shuttle detector - no continual learning complexity.
Used to compare against Twin to see if the context bridge helps shuttle detection.
"""
import os
import sys
from ultralytics import YOLO
from pathlib import Path

def train_shuttle_only(epochs=150):
    """
    Train YOLO11x as a pure shuttle detector (1 class output).
    
    This baseline answers: "Does the Twin's context bridge actually help 
    detect shuttles better than a simple fine-tuned model?"
    """
    print("=" * 60)
    print("E6: Pure Shuttle Detector")
    print("=" * 60)
    print("Config: YOLO11x → 1 class (shuttle only)")
    print("Purpose: Baseline for shuttle detection accuracy")
    print()
    
    project_root = Path(__file__).parent.parent
    
    # Use original shuttle-only dataset (nc=1)
    data_yaml = project_root / "data" / "data.yaml"
    
    print(f"Loading pretrained YOLO11x...")
    model = YOLO("yolo11x.pt")
    
    print(f"Training on: {data_yaml}")
    print("Note: Model will be fine-tuned from COCO weights but output only 1 class")
    
    model.train(
        data=str(data_yaml),
        epochs=epochs,
        imgsz=640,
        batch=16,
        project=str(project_root / "runs" / "detect"),
        name="e6_shuttle_only",
        exist_ok=True,
        device=0,
        workers=8,
        patience=50,
        optimizer='AdamW',
        lr0=0.001,
    )
    
    print("\n>>> E6 Training Complete <<<")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=150)
    args = parser.parse_args()
    
    train_shuttle_only(epochs=args.epochs)
