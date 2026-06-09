"""
E3: Learning without Forgetting (LwF) using Ultralytics Callbacks
Uses the standard YOLO training with a callback to inject distillation loss.
"""
import os
import torch
import torch.nn.functional as F
from ultralytics import YOLO
from ultralytics.utils import callbacks
from pathlib import Path


class LwFCallback:
    """Callback to add distillation loss during training."""
    
    def __init__(self, teacher_model, alpha=10.0):
        self.teacher = teacher_model
        self.alpha = alpha
        self.step_count = 0
        
    def on_train_batch_end(self, trainer):
        """Add distillation loss after each batch."""
        # Access the current batch images
        if not hasattr(trainer, 'batch') or trainer.batch is None:
            return
            
        imgs = trainer.batch['img']
        
        # Get student predictions (already computed by trainer)
        # We need to recompute for distillation
        with torch.no_grad():
            # Teacher forward
            teacher_out = self.teacher(imgs)
            
        # Get student model output
        student = trainer.model
        student_out = student(imgs)
        
        # Compute distillation loss
        if isinstance(teacher_out, tuple):
            teacher_feats = teacher_out[1] if len(teacher_out) > 1 else [teacher_out[0]]
        else:
            teacher_feats = teacher_out if isinstance(teacher_out, list) else [teacher_out]
            
        if isinstance(student_out, tuple):
            student_feats = student_out[1] if len(student_out) > 1 else [student_out[0]]
        else:
            student_feats = student_out if isinstance(student_out, list) else [student_out]
        
        distill_loss = torch.tensor(0.0, device=imgs.device)
        num_matched = 0
        
        for sf, tf in zip(student_feats, teacher_feats):
            if not isinstance(sf, torch.Tensor) or not isinstance(tf, torch.Tensor):
                continue
            if sf.shape[1] != tf.shape[1]:
                min_c = min(sf.shape[1], tf.shape[1])
                sf = sf[:, :min_c, ...]
                tf = tf[:, :min_c, ...]
            if sf.shape[-2:] != tf.shape[-2:]:
                continue
            distill_loss = distill_loss + F.mse_loss(sf, tf)
            num_matched += 1
        
        if num_matched > 0:
            distill_loss = distill_loss / num_matched * self.alpha
            # Add to gradients (backward on additional loss)
            distill_loss.backward()
            
        self.step_count += 1
        if self.step_count == 1 or self.step_count % 500 == 0:
            print(f"[LwF] Step {self.step_count}: Distill_loss={distill_loss.item():.4f}, Matched={num_matched}")


def train_lwf(epochs=150, alpha=10.0):
    """Train with LwF using callbacks."""
    print(f">>> E3: LwF Training with Callbacks (alpha={alpha}) <<<")
    
    project_root = Path(__file__).parent.parent
    
    # Load teacher (frozen)
    print("Loading teacher model...")
    teacher = YOLO("yolo11x.pt").model
    teacher.eval()
    for p in teacher.parameters():
        p.requires_grad = False
    teacher.cuda()
    print(f"Teacher ready: {sum(p.numel() for p in teacher.parameters())} params")
    
    # Create LwF callback
    lwf_callback = LwFCallback(teacher, alpha=alpha)
    
    # Load student and train
    print("Starting student training...")
    student = YOLO("yolo11x.pt")
    
    # Add callback
    student.add_callback("on_train_batch_end", lwf_callback.on_train_batch_end)
    
    # Train
    student.train(
        data=str(project_root / "data" / "data_81class.yaml"),
        epochs=epochs,
        imgsz=640,
        batch=16,
        project=str(project_root / "runs" / "detect"),
        name="e3_lwf",
        exist_ok=True,
        device=0,
        workers=8,
        patience=50,
        optimizer='AdamW',
        lr0=0.0005,
    )
    
    print(">>> E3 LwF Complete <<<")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=150)
    parser.add_argument('--alpha', type=float, default=10.0)
    args = parser.parse_args()
    
    train_lwf(epochs=args.epochs, alpha=args.alpha)
