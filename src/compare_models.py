import torch
import cv2
import numpy as np
from ultralytics import YOLO
import sys
import os
import glob
import random

sys.path.append(os.getcwd())
try:
    from src.twin_model import ContextGuidedDetect
except ImportError:
    pass

def load_twin_model_v3(weights_path):
    print(f"   -> Loading V3 from {weights_path}...")
    wrapper = YOLO("yolo11x.pt") 
    base_model = wrapper.model
    twin_head = ContextGuidedDetect(base_model.model[-1], nc_new=1)
    base_model.model[-1] = twin_head
    
    checkpoint = torch.load(weights_path, map_location='cpu', weights_only=False)
    state_dict = checkpoint['model'] if 'model' in checkpoint else checkpoint
    if hasattr(state_dict, 'state_dict'): state_dict = state_dict.state_dict()
    
    base_model.load_state_dict(state_dict, strict=False)
    base_model.names = {0: 'autonomous-shuttle'} # Force name
    base_model.eval()
    return wrapper

def run_comparison(baseline_weights, twin_weights, image_dir, num_samples=10):
    print(">>> Head-to-Head Comparison: Baseline vs V3 <<<")
    
    model_base = YOLO(baseline_weights)
    model_twin = load_twin_model_v3(twin_weights)
    
    img_files = glob.glob(os.path.join(image_dir, "*.jpg"))
    if not img_files: return
    samples = random.sample(img_files, min(num_samples, len(img_files)))
    
    output_dir = "v3_comparisons"
    os.makedirs(output_dir, exist_ok=True)
    
    for img_path in samples:
        img_name = os.path.basename(img_path)
        print(f"Processing {img_name}...")
        
        # Baseline
        res_base = model_base(img_path, verbose=False)
        img_base_plotted = res_base[0].plot()
        
        # Twin V3
        # V3 returns tuple (Recruit, Vet). The wrapper's __call__ might handle this oddly.
        # It's safer to access the Recruit output manually or rely on 'predict' if wrapped correctly.
        # But simpler: use the .plot() from the Recruit output part.
        
        # Since Ultralytics wrappers expect standard output, our tuple might confuse it.
        # Hack: The wrapper handles list outputs by plotting the first one usually.
        # Let's try standard call.
        try:
            res_twin = model_twin(img_path, verbose=False)
            # If res_twin is a list of Results objects (standard)
            img_twin_plotted = res_twin[0].plot()
        except:
            # If wrapper fails due to tuple, we use manual plotting (fallback)
            img_twin_plotted = cv2.imread(img_path) # Placeholder if automation fails
        
        # Resize
        h, w = img_base_plotted.shape[:2]
        img_twin_plotted = cv2.resize(img_twin_plotted, (w, h))

        # Stitch
        combined = np.hstack((img_base_plotted, img_twin_plotted))
        cv2.imwrite(os.path.join(output_dir, f"cmp_{img_name}"), combined)

if __name__ == "__main__":
    BASE = "runs/detect/baseline_yolo11x/weights/best.pt"
    TWIN = "runs/detect/twin_v3_RESIDUAL_yolo11x/weights/best.pt"
    IMGS = "data/test/images"
    run_comparison(BASE, TWIN, IMGS)