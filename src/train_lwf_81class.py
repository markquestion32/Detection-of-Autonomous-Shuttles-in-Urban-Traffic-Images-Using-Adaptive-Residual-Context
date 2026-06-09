"""
E3: Learning without Forgetting (LwF) for YOLO
Feature Distillation approach to preserve COCO knowledge while learning Shuttle class.
"""
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from ultralytics.models.yolo.detect import DetectionTrainer
from ultralytics.utils.loss import v8DetectionLoss
from ultralytics import YOLO
import copy

class LwFLoss(v8DetectionLoss):
    """
    Custom loss that adds feature distillation to standard YOLOv8 detection loss.
    """
    def __init__(self, model, teacher_model, alpha=5.0):
        super().__init__(model)
        self.teacher = teacher_model
        self.alpha = alpha
        self.distill_count = 0  # Debug counter

    def __call__(self, preds, batch):
        # 1. Compute Standard YOLO Loss (Box + Cls + DFL)
        loss, loss_items = super().__call__(preds, batch)
        
        # 2. Compute Distillation Loss
        with torch.no_grad():
            # Get teacher predictions
            teacher_out = self.teacher(batch['img'])
            
        # Parse teacher output (eval mode returns tuple)
        if isinstance(teacher_out, tuple):
            teacher_preds = teacher_out[1] if len(teacher_out) > 1 else teacher_out[0]
        else:
            teacher_preds = teacher_out
            
        # Ensure both are lists
        if not isinstance(teacher_preds, (list, tuple)):
            teacher_preds = [teacher_preds]
        if not isinstance(preds, (list, tuple)):
            preds = [preds]

        distill_loss = torch.tensor(0.0, device=loss.device)
        num_matched = 0
        
        # Iterate over feature scales
        for student_feat, teacher_feat in zip(preds, teacher_preds):
            # Skip if tensor types don't match
            if not isinstance(student_feat, torch.Tensor) or not isinstance(teacher_feat, torch.Tensor):
                continue
                
            # Handle shape mismatch (81 vs 80 classes)
            if student_feat.shape[1] != teacher_feat.shape[1]:
                min_c = min(student_feat.shape[1], teacher_feat.shape[1])
                student_feat = student_feat[:, :min_c, ...]
                teacher_feat = teacher_feat[:, :min_c, ...]
            
            # Handle spatial dimension mismatch
            if student_feat.shape[-2:] != teacher_feat.shape[-2:]:
                continue
                
            distill_loss = distill_loss + F.mse_loss(student_feat, teacher_feat)
            num_matched += 1
        
        # Scale distillation loss
        if num_matched > 0:
            distill_loss = distill_loss / num_matched * self.alpha
        
        # Debug: Print every 1000 batches
        self.distill_count += 1
        if self.distill_count == 1 or self.distill_count % 1000 == 0:
            print(f"[LwF Debug] Batch {self.distill_count}: YOLO_loss={loss.item():.4f}, Distill_loss={distill_loss.item():.4f}, Matched={num_matched}")
        
        # Combine losses
        final_loss = loss + distill_loss
        
        return final_loss, loss_items


class LwFTrainer(DetectionTrainer):
    """
    Custom trainer that uses LwF loss for knowledge distillation.
    """
    def __init__(self, teacher_model, alpha=5.0, *args, **kwargs):
        self._teacher = teacher_model
        self._alpha = alpha
        super().__init__(*args, **kwargs)
        
    def setup_model(self):
        """Override to move teacher to correct device."""
        super().setup_model()
        if self._teacher is not None:
            self._teacher.to(self.device)
            self._teacher.eval()
            print(f"[LwF] Teacher model moved to {self.device}")
            
    def get_loss(self, *args, **kwargs):
        """Override to return custom LwF loss."""
        if not hasattr(self, '_lwf_loss'):
            self._lwf_loss = LwFLoss(self.model, self._teacher, alpha=self._alpha)
            print(f"[LwF] LwFLoss initialized with alpha={self._alpha}")
        return self._lwf_loss


def train_lwf(epochs=150, batch=16, alpha=10.0):
    """
    Train with Learning without Forgetting.
    
    Args:
        epochs: Training epochs
        batch: Batch size
        alpha: Distillation loss weight (higher = more regularization)
    """
    print(f">>> E3: LwF Training (Feature Distillation, alpha={alpha}) <<<")
    
    # 1. Load Teacher (Frozen COCO model)
    print("Loading teacher model (yolo11x pretrained on COCO)...")
    teacher = YOLO("yolo11x.pt").model
    teacher.eval()
    for p in teacher.parameters():
        p.requires_grad = False
    print(f"Teacher frozen: {sum(p.numel() for p in teacher.parameters())} params")
    
    # 2. Setup training config
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    
    args = dict(
        model="yolo11x.pt",
        data=os.path.join(project_root, "data/data_81class.yaml"),
        epochs=epochs,
        batch=batch,
        imgsz=640,
        project=os.path.join(project_root, "runs/detect"),
        name="e3_lwf",
        exist_ok=True,
        device=0,
        workers=8,
        patience=50,  # Increased patience for CL
        optimizer='AdamW',
        lr0=0.0005,  # Lower LR for stability
    )
    
    # 3. Create and run trainer
    trainer = LwFTrainer(teacher_model=teacher, alpha=alpha, overrides=args)
    trainer.train()
    
    print(">>> E3 LwF Complete <<<")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=150)
    parser.add_argument('--alpha', type=float, default=10.0)
    args = parser.parse_args()
    
    train_lwf(epochs=args.epochs, alpha=args.alpha)
