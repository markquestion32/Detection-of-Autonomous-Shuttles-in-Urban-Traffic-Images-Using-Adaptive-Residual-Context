import sys
import os
from ultralytics import YOLO
import torch
import torch.nn as nn

# --- 1. ROBUST PATH SETUP ---
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)
if project_root not in sys.path:
    sys.path.append(project_root)

DATA_PATH = os.path.join(project_root, "data/data.yaml")

# Import Architecture
try:
    from src.twin_model import ContextGuidedDetect
except ImportError:
    sys.path.append(script_dir)
    from twin_model import ContextGuidedDetect

# --- 2. THE SURGERY FUNCTION ---
def attach_v3_head(model):
    print(">>> PRE-TRAIN SURGERY: Swapping Head for Twin Architecture (V3) <<<")
    
    # Access the internal PyTorch model
    # Note: Ultralytics wraps it in .model
    if not hasattr(model, 'model') or model.model is None:
        # Force load if not loaded yet
        _ = model.predict(torch.zeros(1, 3, 640, 640), verbose=False)
    
    base_model_layers = model.model.model 
    
    # Locate the Detect Head
    det_head_idx = -1
    for i, m in enumerate(base_model_layers):
        if m.__class__.__name__ == 'Detect':
             det_head_idx = i
             break
                 
    if det_head_idx == -1: 
        raise ValueError("Could not find Detect head")
    else:
        print(f"   -> Found Detect head at index {det_head_idx}")

    # Swap
    old_head = base_model_layers[det_head_idx]
    
    # Create V3 Head
    new_head = ContextGuidedDetect(old_head, nc_new=1) # 1 Class: Shuttle
    
    # Ensure it's on the right device
    device = next(old_head.parameters()).device
    new_head.to(device)
    
    # Inject
    base_model_layers[det_head_idx] = new_head
    
    # CRITICAL: Update model config overrides to prevent errors during saving
    model.model.nc = 1
    model.model.names = {0: 'autonomous-shuttle'}
    
    print(">>> SURGERY COMPLETE: Twin Head V3 Attached (Before EMA Init) <<<")
    return model

# --- 3. Main Training Block ---
if __name__ == '__main__':
    # 1. Load Skeleton
    print("Loading Baseline...")
    model = YOLO("yolo11x.pt")
    
    # 2. Perform Surgery
    model = attach_v3_head(model)
    
    # 3. Train
    # WARNING: calling model.train() often triggers a reload if 'model' arg is passed in yaml or defaults.
    # To prevent this, we must ensure the trainer uses OUR model object.
    
    print("Starting Training (Direct Memory Object)...")
    
    # We pass the model object itself. 
    # Ultralytics should respect the in-memory model if we don't pass a 'model=' kwarg that conflicts.
    # However, 'model.train()' methods sometimes default to self.model.
    
    # Let's verify the surgery one last time before training
    print("Verifying Surgery Persistence...")
    print(model.model.model[-1])
    
    # CRITICAL FIX REVISED V2:
    # use DetectionTrainer directly to support custom in-memory model
    # CRITICAL FIX REVISED V3:
    # use Custom subclass to bypass model loading logic entirely
    # CRITICAL FIX: Direct override of the trainer logic
    from ultralytics.models.yolo.detect import DetectionTrainer
    
    class CustomTrainer(DetectionTrainer):
        def get_model(self, cfg=None, weights=None, verbose=True):
            print(">>> CustomTrainer: Returning in-memory twin model (FORCED) <<<")
            # We strictly return the custom model we attached
            # Re-attaching the head just to be paranoid about object references
            return self.custom_model

        def _setup_train(self):
            """
            Override setup to aggressively re-freeze the Veteran head
            after Ultralytics tries to unfreeze everything.
            """
            # 1. Run standard setup (which triggers the unfreeze warnings)
            super()._setup_train()
            
            # 2. Re-freeze Veteran Head
            print(">>> CustomTrainer: Re-freezing Veteran Head (ContextGuidedDetect) <<<")
            found_twin = False
            
            # Traverse the model to find our custom head
            # trainer.model is the DetectionModel wrapper
            if hasattr(self.model, 'model'): # accessing the nn.Sequential
                for module in self.model.model:
                    if module.__class__.__name__ == 'ContextGuidedDetect':
                        print("   -> Found ContextGuidedDetect! Freezing context_head...")
                        for name, p in module.context_head.named_parameters():
                            p.requires_grad = False
                        
                        # Verify
                        grad_status = [p.requires_grad for p in module.context_head.parameters()]
                        if any(grad_status):
                            print("   ⚠️ WARNING: Some veteran params are still trainable!")
                        else:
                            print("   -> SUCCESS: All veteran params frozen.")
                        found_twin = True
                        break
            
            if not found_twin:
                 print("   ⚠️ WARNING: Could not find ContextGuidedDetect module to freeze!")


    print("Initializing CustomTrainer...")
    
    # We remove 'model' from overrides so it doesn't try to load from disk
    # Ultralytics checks args.model or cfg.model usually.
    overrides = {
        'task': 'detect',
        'mode': 'train',
        'model': "yolo11x.pt", # Dummy string to satisfy __init__ check
        'data': DATA_PATH,
        'epochs': 150,
        'imgsz': 640,
        'batch': 16,
        'workers': 4,
        'name': "twin",
        'optimizer': "AdamW",
        'lr0': 0.001,
        'patience': 25,
        'save': True,
        'device': 0,
        'amp': False,
        'exist_ok': False,
    }
    
    # We must instantiate with a dummy model to pass initial checks, 
    # but we rely on get_model to swap it out.
    trainer = CustomTrainer(overrides=overrides)
    
    # INJECT CUSTOM MODEL
    trainer.custom_model = model.model # The nn.Module
    
    # Force the trainer's model attribute to be our custom one immediately
    trainer.model = trainer.custom_model

    # Train
    print("Starting Training (CustomTrainer)...")
    trainer.train()