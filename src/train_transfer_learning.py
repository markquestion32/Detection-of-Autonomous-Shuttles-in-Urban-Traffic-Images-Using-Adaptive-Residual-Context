"""
E5: Transfer Learning with Surgical Freezing
Freeze backbone + Freeze COCO class logits + Train ONLY shuttle logit

This is the MINIMUM viable solution for preserving COCO while learning shuttle.

Key insight: We must prevent gradients from modifying COCO class logits.
"""
import os
import sys
import torch
import torch.nn as nn
from pathlib import Path
from ultralytics import YOLO


def expand_head_for_new_class(model, nc_new=81):
    """
    Expand the detection head from 80 classes to 81 classes.
    Copies COCO weights and adds new shuttle class.
    """
    detect = model.model[-1]
    nl = detect.nl  # Number of detection layers (3 for P3, P4, P5)
    nc_old = detect.nc  # Original number of classes (80)
    
    print(f"Expanding head from {nc_old} to {nc_new} classes")
    
    for i in range(nl):
        old_conv = detect.cv3[i][-1]
        
        # Create new conv with nc_new output channels
        new_conv = nn.Conv2d(
            old_conv.in_channels, 
            nc_new, 
            kernel_size=1, 
            stride=1, 
            padding=0
        )
        
        # Copy old weights for existing COCO classes
        with torch.no_grad():
            new_conv.weight[:nc_old] = old_conv.weight
            new_conv.bias[:nc_old] = old_conv.bias
            
            # Initialize new shuttle class weights
            nn.init.normal_(new_conv.weight[nc_old:], std=0.01)
            nn.init.constant_(new_conv.bias[nc_old:], -4.0)  # Low initial confidence
        
        new_conv = new_conv.to(old_conv.weight.device)
        detect.cv3[i][-1] = new_conv
        print(f"  Head {i}: {nc_old} -> {nc_new} classes")
    
    # Update detection module attributes
    detect.nc = nc_new
    detect.no = nc_new + detect.reg_max * 4
    
    return model


def surgical_freeze(model):
    """
    Surgical Freezing Strategy:
    1. Freeze entire backbone (feature extraction)
    2. Freeze box regression (cv2)
    3. Freeze COCO class logits (channels 0-79 in cv3)
    4. Train ONLY shuttle class logit (channel 80 in cv3)
    """
    print("\n=== Surgical Freeze ===")
    
    # 1. Freeze EVERYTHING first
    for param in model.model.parameters():
        param.requires_grad = False
    print("  [✓] Froze all parameters")
    
    # 2. Unfreeze ONLY shuttle class weights (channel 80) in cv3
    detect = model.model[-1]  # Detection head
    
    for i in range(detect.nl):
        conv = detect.cv3[i][-1]  # Last conv in class prediction branch
        
        # Freeze COCO class weights (channels 0-79)
        conv.weight[:80].requires_grad = False
        conv.bias[:80].requires_grad = False
        
        # Train shuttle class only (channel 80)
        conv.weight[80:].requires_grad = True
        conv.bias[80:].requires_grad = True
        
        print(f"  [✓] Head {i}: Froze COCO (0-79), Training Shuttle (80)")
    
    # Count parameters
    frozen = sum(p.numel() for p in model.model.parameters() if not p.requires_grad)
    trainable = sum(p.numel() for p in model.model.parameters() if p.requires_grad)
    
    print(f"\n  Frozen parameters: {frozen:,}")
    print(f"  Trainable parameters: {trainable:,}")
    print(f"  Trainable ratio: {trainable/frozen*100:.4f}%")
    
    return model


def train_surgical_transfer(epochs=150):
    """
    Train with Surgical Freezing:
    - Freeze backbone (preserves features)
    - Freeze COCO logits (preserves COCO predictions)
    - Train ONLY shuttle logit (learns shuttle)
    """
    print("=" * 60)
    print("E5: Transfer Learning with Surgical Freezing")
    print("=" * 60)
    print("Strategy: Freeze backbone + COCO logits, train ONLY shuttle logit")
    print("Goal: Preserve COCO + Learn Shuttle with minimal parameter changes")
    print()
    
    project_root = Path(__file__).parent.parent
    
    # 1. Load pretrained YOLO
    print("Loading pretrained YOLO11x...")
    model = YOLO("yolo11x.pt")
    
    # 2. Expand head to 81 classes (keeping COCO weights)
    expand_head_for_new_class(model.model, nc_new=81)
    
    # 3. Apply surgical freeze
    surgical_freeze(model.model)
    
    # 4. Train on 81-class dataset
    data_yaml = project_root / "data" / "data_81class.yaml"
    
    print(f"\nUsing dataset: {data_yaml}")
    print("\nStarting training (only shuttle logit will update)...")
    
    # Note: We set freeze=None to use our custom freeze
    # The optimizer will only update requires_grad=True params
    model.train(
        data=str(data_yaml),
        epochs=epochs,
        imgsz=640,
        batch=16,
        project=str(project_root / "runs" / "detect"),
        name="e5_transfer",
        exist_ok=True,
        device=0,
        workers=8,
        patience=50,
        optimizer='AdamW',
        lr0=0.01,  # Higher LR since we're training very few params
    )
    
    print("\n>>> E5 Surgical Transfer Learning Complete <<<")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=150)
    args = parser.parse_args()
    
    train_surgical_transfer(epochs=args.epochs)
