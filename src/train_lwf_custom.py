"""
E3: Learning without Forgetting (LwF) - Custom Training Loop
This implements LwF with a fully custom PyTorch training loop,
bypassing Ultralytics Trainer to ensure distillation loss is applied.
"""
import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from ultralytics import YOLO
from ultralytics.utils.loss import v8DetectionLoss
from pathlib import Path
from tqdm import tqdm
import yaml

def train_lwf_custom(epochs=150, batch=16, alpha=10.0, lr=0.0005):
    """
    Custom LwF training loop with explicit distillation loss.
    """
    print(f">>> E3: LwF Custom Training (alpha={alpha}, lr={lr}) <<<")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    project_root = Path(__file__).parent.parent
    
    # === 1. Load Teacher (Frozen) ===
    print("Loading teacher model...")
    teacher_wrapper = YOLO("yolo11x.pt")
    teacher = teacher_wrapper.model.to(device)
    teacher.eval()
    for p in teacher.parameters():
        p.requires_grad = False
    print(f"Teacher: {sum(p.numel() for p in teacher.parameters())} params (frozen)")
    
    # === 2. Load Student ===
    print("Loading student model...")
    student_wrapper = YOLO("yolo11x.pt")
    student = student_wrapper.model.to(device)
    student.train()
    print(f"Student: {sum(p.numel() for p in student.parameters())} params (trainable)")
    
    # === 3. Setup Data ===
    data_yaml = project_root / "data" / "data_81class.yaml"
    
    # Load data config
    import yaml
    with open(data_yaml, 'r') as f:
        data_cfg = yaml.safe_load(f)
    
    # Get training images path
    train_path = Path(data_cfg.get('path', '')) / data_cfg.get('train', 'train/images')
    if not train_path.is_absolute():
        train_path = project_root / "data" / "data_81class" / data_cfg.get('train', 'train/images')
    
    # Use Ultralytics data loading via YOLO wrapper
    from ultralytics.data.utils import check_det_dataset
    from ultralytics.data.dataset import YOLODataset
    
    # Check and load dataset
    data_dict = check_det_dataset(str(data_yaml))
    
    # Default hyperparameters for augmentation
    default_hyp = {
        'mosaic': 1.0,
        'mixup': 0.0,
        'copy_paste': 0.0,
        'degrees': 0.0,
        'translate': 0.1,
        'scale': 0.5,
        'shear': 0.0,
        'perspective': 0.0,
        'flipud': 0.0,
        'fliplr': 0.5,
        'bgr': 0.0,
        'hsv_h': 0.015,
        'hsv_s': 0.7,
        'hsv_v': 0.4,
        'erasing': 0.4,
    }
    
    # Create a simple namespace-like object for hyp
    class HypConfig:
        def __init__(self, d):
            for k, v in d.items():
                setattr(self, k, v)
    
    hyp = HypConfig(default_hyp)
    
    dataset = YOLODataset(
        img_path=data_dict['train'],
        imgsz=640,
        batch_size=batch,
        augment=False,  # Disable augmentation for simplicity
        hyp=hyp,
        rect=False,
        cache=False,
        single_cls=False,
        stride=32,
        pad=0.5,
        data=data_dict
    )
    
    dataloader = DataLoader(
        dataset,
        batch_size=batch,
        shuffle=True,
        num_workers=8,
        collate_fn=getattr(dataset, 'collate_fn', None),
        pin_memory=True
    )
    
    print(f"Dataset: {len(dataset)} images, {len(dataloader)} batches")
    
    # === 4. Setup Loss & Optimizer ===
    yolo_loss = v8DetectionLoss(student)
    optimizer = torch.optim.AdamW(student.parameters(), lr=lr, weight_decay=0.0005)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    
    # === 5. Training Loop ===
    save_dir = project_root / "runs" / "detect" / "e3_lwf"
    save_dir.mkdir(parents=True, exist_ok=True)
    
    best_loss = float('inf')
    
    for epoch in range(epochs):
        student.train()
        epoch_loss = 0.0
        epoch_yolo_loss = 0.0
        epoch_distill_loss = 0.0
        
        pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{epochs}")
        for batch_idx, batch_data in enumerate(pbar):
            # Move to device
            imgs = batch_data['img'].to(device).float() / 255.0
            
            # Forward pass - Student
            student_out = student(imgs)
            
            # Compute YOLO loss
            yolo_l, _ = yolo_loss(student_out, batch_data)
            
            # Forward pass - Teacher (no grad)
            with torch.no_grad():
                teacher_out = teacher(imgs)
            
            # Compute Distillation Loss
            # Extract features (student_out is list of [B, C, H, W] tensors)
            if isinstance(teacher_out, tuple):
                teacher_feats = teacher_out[1] if len(teacher_out) > 1 else [teacher_out[0]]
            else:
                teacher_feats = teacher_out if isinstance(teacher_out, list) else [teacher_out]
                
            student_feats = student_out if isinstance(student_out, list) else [student_out]
            
            distill_loss = torch.tensor(0.0, device=device)
            num_matched = 0
            
            for sf, tf in zip(student_feats, teacher_feats):
                if not isinstance(sf, torch.Tensor) or not isinstance(tf, torch.Tensor):
                    continue
                    
                # Handle channel mismatch (81 vs 80)
                if sf.shape[1] != tf.shape[1]:
                    min_c = min(sf.shape[1], tf.shape[1])
                    sf = sf[:, :min_c, ...]
                    tf = tf[:, :min_c, ...]
                
                # Handle spatial mismatch
                if sf.shape[-2:] != tf.shape[-2:]:
                    continue
                    
                distill_loss = distill_loss + F.mse_loss(sf, tf)
                num_matched += 1
            
            if num_matched > 0:
                distill_loss = distill_loss / num_matched * alpha
            
            # Total loss
            total_loss = yolo_l + distill_loss
            
            # Backward
            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()
            
            # Track
            epoch_loss += total_loss.item()
            epoch_yolo_loss += yolo_l.item()
            epoch_distill_loss += distill_loss.item()
            
            pbar.set_postfix({
                'yolo': f'{yolo_l.item():.3f}',
                'distill': f'{distill_loss.item():.3f}',
                'total': f'{total_loss.item():.3f}'
            })
        
        scheduler.step()
        
        avg_loss = epoch_loss / len(dataloader)
        avg_yolo = epoch_yolo_loss / len(dataloader)
        avg_distill = epoch_distill_loss / len(dataloader)
        
        print(f"Epoch {epoch+1}: Total={avg_loss:.4f}, YOLO={avg_yolo:.4f}, Distill={avg_distill:.4f}")
        
        # Save checkpoint
        if avg_loss < best_loss:
            best_loss = avg_loss
            ckpt = {
                'epoch': epoch,
                'model': student.state_dict(),
                'optimizer': optimizer.state_dict(),
                'best_loss': best_loss
            }
            torch.save(ckpt, save_dir / "best.pt")
            print(f"  Saved best model (loss={best_loss:.4f})")
        
        # Save last
        ckpt = {
            'epoch': epoch,
            'model': student.state_dict(),
            'optimizer': optimizer.state_dict(),
        }
        torch.save(ckpt, save_dir / "last.pt")
    
    print(f">>> E3 LwF Training Complete <<<")
    print(f"Best model saved to: {save_dir / 'best.pt'}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=150)
    parser.add_argument('--alpha', type=float, default=10.0)
    parser.add_argument('--lr', type=float, default=0.0005)
    args = parser.parse_args()
    
    train_lwf_custom(epochs=args.epochs, alpha=args.alpha, lr=args.lr)
