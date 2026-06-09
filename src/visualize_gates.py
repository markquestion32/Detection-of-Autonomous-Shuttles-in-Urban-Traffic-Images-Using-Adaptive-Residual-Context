import torch
import cv2
import numpy as np
import matplotlib.pyplot as plt
from ultralytics import YOLO
import sys
import os

sys.path.append(os.getcwd())
try:
    from src.twin_model import ContextGuidedDetect
except ImportError:
    pass

def visualize_attention(weights_path, image_path):
    print(f"Visualizing V3 Gates for {image_path}...")
    
    wrapper = YOLO("yolo11x.pt")
    base_model = wrapper.model
    twin_head = ContextGuidedDetect(base_model.model[-1], nc_new=1)
    base_model.model[-1] = twin_head
    
    checkpoint = torch.load(weights_path, map_location='cpu', weights_only=False)
    state_dict = checkpoint['model'] if 'model' in checkpoint else checkpoint
    if hasattr(state_dict, 'state_dict'): state_dict = state_dict.state_dict()
    base_model.load_state_dict(state_dict, strict=False)
    base_model.eval()
    
    img = cv2.imread(image_path)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_t = cv2.resize(img_rgb, (640, 640))
    img_t = torch.from_numpy(img_t).float().permute(2,0,1).unsqueeze(0) / 255.0
    
    activations = []
    def hook_fn(module, input, output):
        activations.append(output.detach().cpu())

    # V3: Hook into 'spatial_gates'
    hooks = []
    for i in range(3):
        h = twin_head.spatial_gates[i][-1].register_forward_hook(hook_fn)
        hooks.append(h)
        
    base_model(img_t)
    
    for h in hooks: h.remove()
    
    plt.figure(figsize=(15, 5))
    
    plt.subplot(1, 4, 1)
    plt.imshow(img_rgb)
    plt.title("Input")
    plt.axis('off')
    
    # Create overlays
    alpha = 0.6  # Transparency for the heatmap
    
    for i in range(3):
        gate_map = activations[i][0, 0].numpy()
        
        # Smooth upsampling using cubic interpolation
        gate_map = cv2.resize(gate_map, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_CUBIC)
        
        # Normalize to 0-255 for color mapping
        gate_map_norm = np.uint8(255 * np.clip(gate_map, 0, 1))
        
        # Apply colormap (Jet)
        heatmap = cv2.applyColorMap(gate_map_norm, cv2.COLORMAP_JET)
        heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
        
        # Blend with original image
        overlay = cv2.addWeighted(heatmap, alpha, img_rgb, 1 - alpha, 0)
        
        plt.subplot(1, 4, i+2)
        plt.imshow(overlay)
        plt.title(f"Gate P{i+3} (Overlay)")
        plt.axis('off')
        
    plt.tight_layout()
    plt.savefig(f"v3_gates_{os.path.basename(image_path)}")
    print("Saved visualization.")

if __name__ == "__main__":
    W = "runs/detect/twin_v3_RESIDUAL_yolo11x8/weights/best.pt" 
    IMG = "data/valid/images/10_jpg.rf.eddd70022caad18099d1dbe8dcd292fa.jpg" # Update this to a real path
    visualize_attention(W, IMG)