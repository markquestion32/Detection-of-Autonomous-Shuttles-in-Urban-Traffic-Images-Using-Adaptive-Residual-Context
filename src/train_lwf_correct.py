"""
E3: Learning without Forgetting (LwF) - Correct Implementation
Based on the original LwF paper implementation.

Key insight: LwF uses Knowledge Distillation on SOFTMAX OUTPUTS, not raw features.
The distillation loss is: KL(softmax(student/T) || softmax(teacher/T))
"""
import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path
from tqdm import tqdm
import copy

from ultralytics import YOLO
from ultralytics.models.yolo.detect import DetectionTrainer


def knowledge_distillation_loss(student_logits, teacher_logits, temperature=2.0):
    """
    Compute Knowledge Distillation loss (soft target loss).
    
    This is the core of LwF: force student to produce similar class probabilities as teacher.
    
    Args:
        student_logits: Student model outputs (before softmax)
        teacher_logits: Teacher model outputs (before softmax)  
        temperature: Softening temperature (higher = softer distributions)
    
    Returns:
        KL divergence loss between softened distributions
    """
    # Soften with temperature
    student_soft = F.log_softmax(student_logits / temperature, dim=1)
    teacher_soft = F.softmax(teacher_logits / temperature, dim=1)
    
    # KL divergence (equivalent to cross-entropy with soft labels)
    loss = F.kl_div(student_soft, teacher_soft, reduction='batchmean') * (temperature ** 2)
    
    return loss


class LwFDetectionLoss:
    """
    Custom loss that combines YOLO detection loss with Knowledge Distillation.
    
    Unlike our previous approach (feature MSE), this uses the correct LwF formulation:
    distilling the CLASS PREDICTIONS (softmax outputs).
    """
    def __init__(self, student_model, teacher_model, base_loss, alpha=1.0, temperature=2.0):
        self.student = student_model
        self.teacher = teacher_model
        self.base_loss = base_loss
        self.alpha = alpha
        self.temperature = temperature
        self.call_count = 0
        
    def __call__(self, preds, batch):
        """
        Compute combined loss: YOLO_loss + alpha * KD_loss
        
        For detection, we apply KD to the class prediction channels of each detection head.
        """
        # 1. Compute standard YOLO detection loss
        base_loss, loss_items = self.base_loss(preds, batch)
        
        # 2. Get teacher predictions (on same input batch)
        with torch.no_grad():
            teacher_out = self.teacher(batch['img'])
            
        # Parse outputs
        if isinstance(teacher_out, tuple):
            teacher_preds = teacher_out[1] if len(teacher_out) > 1 else teacher_out[0]
        else:
            teacher_preds = teacher_out
            
        if not isinstance(teacher_preds, (list, tuple)):
            teacher_preds = [teacher_preds]
        if not isinstance(preds, (list, tuple)):
            preds = [preds]
            
        # 3. Compute Knowledge Distillation loss
        kd_loss = torch.tensor(0.0, device=base_loss.device)
        num_matched = 0
        
        for student_feat, teacher_feat in zip(preds, teacher_preds):
            if not isinstance(student_feat, torch.Tensor) or not isinstance(teacher_feat, torch.Tensor):
                continue
                
            # YOLO detection head output: [B, 4+num_classes, H, W] or [B, num_channels, num_anchors]
            # We want to distill the CLASS channels only (skip box regression)
            
            # Get number of classes in each (teacher has 80, student has 81)
            s_nc = student_feat.shape[1] - 4  # Assuming first 4 are box coords
            t_nc = teacher_feat.shape[1] - 4
            
            if s_nc <= 0 or t_nc <= 0:
                continue
                
            # Extract class logits (skip first 4 box channels)
            student_cls = student_feat[:, 4:4+min(s_nc, t_nc), ...]
            teacher_cls = teacher_feat[:, 4:4+min(s_nc, t_nc), ...]
            
            # Handle spatial dimension mismatch
            if student_cls.shape != teacher_cls.shape:
                continue
            
            # Reshape for KD: [B, C, H, W] -> [B*H*W, C]
            B, C = student_cls.shape[:2]
            student_flat = student_cls.reshape(B, C, -1).permute(0, 2, 1).reshape(-1, C)
            teacher_flat = teacher_cls.reshape(B, C, -1).permute(0, 2, 1).reshape(-1, C)
            
            # Compute KD loss
            kd_loss = kd_loss + knowledge_distillation_loss(
                student_flat, teacher_flat, self.temperature
            )
            num_matched += 1
        
        # Scale KD loss
        if num_matched > 0:
            kd_loss = kd_loss / num_matched * self.alpha
            
        # Debug logging
        self.call_count += 1
        if self.call_count == 1 or self.call_count % 500 == 0:
            print(f"[LwF] Step {self.call_count}: Base={base_loss.item():.4f}, KD={kd_loss.item():.4f}, Matched={num_matched}")
        
        return base_loss + kd_loss, loss_items


class LwFTrainer(DetectionTrainer):
    """Custom trainer that uses LwF loss."""
    
    def __init__(self, teacher_model, alpha=1.0, temperature=2.0, *args, **kwargs):
        self._teacher = teacher_model
        self._alpha = alpha
        self._temperature = temperature
        super().__init__(*args, **kwargs)
        
    def setup_model(self):
        """Move teacher to correct device."""
        super().setup_model()
        if self._teacher is not None:
            self._teacher.to(self.device)
            self._teacher.eval()
            print(f"[LwF] Teacher moved to {self.device}")
            
    def get_loss(self, *args, **kwargs):
        """Return custom LwF loss."""
        if not hasattr(self, '_lwf_loss'):
            from ultralytics.utils.loss import v8DetectionLoss
            base_loss = v8DetectionLoss(self.model)
            self._lwf_loss = LwFDetectionLoss(
                self.model, self._teacher, base_loss,
                alpha=self._alpha, temperature=self._temperature
            )
            print(f"[LwF] Loss initialized: alpha={self._alpha}, T={self._temperature}")
        return self._lwf_loss


def train_lwf(epochs=150, alpha=1.0, temperature=2.0):
    """Train with Learning without Forgetting."""
    print(f">>> E3: LwF Training (alpha={alpha}, T={temperature}) <<<")
    
    project_root = Path(__file__).parent.parent
    
    # Load teacher (frozen COCO model)
    print("Loading teacher model...")
    teacher = YOLO("yolo11x.pt").model
    teacher.eval()
    for p in teacher.parameters():
        p.requires_grad = False
    print(f"Teacher: {sum(p.numel() for p in teacher.parameters())} params (frozen)")
    
    # Setup training
    args = dict(
        model="yolo11x.pt",
        data=str(project_root / "data" / "data_81class.yaml"),
        epochs=epochs,
        batch=16,
        imgsz=640,
        project=str(project_root / "runs" / "detect"),
        name="e3_lwf",
        exist_ok=True,
        device=0,
        workers=8,
        patience=50,
        optimizer='AdamW',
        lr0=0.001,  # Standard LR
    )
    
    # Create and run trainer
    trainer = LwFTrainer(
        teacher_model=teacher, 
        alpha=alpha, 
        temperature=temperature,
        overrides=args
    )
    trainer.train()
    
    print(">>> E3 LwF Complete <<<")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=150)
    parser.add_argument('--alpha', type=float, default=1.0)
    parser.add_argument('--temperature', type=float, default=2.0)
    args = parser.parse_args()
    
    train_lwf(epochs=args.epochs, alpha=args.alpha, temperature=args.temperature)
