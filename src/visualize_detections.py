"""
Generate a 5x5 grid of detection visualizations.
- Green boxes = Autonomous Shuttles
- Red boxes = COCO classes
"""
import os
import sys
import torch
import numpy as np
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont

# Add project root
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ultralytics import YOLO

try:
    from src.twin_model import ContextGuidedDetect
except ImportError:
    from twin_model import ContextGuidedDetect


def load_twin_model(weights_path):
    """Load the Twin architecture model."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    wrapper = YOLO("yolo11x.pt")
    base_model = wrapper.model
    
    # Attach twin head
    twin_head = ContextGuidedDetect(base_model.model[-1], nc_new=1)
    twin_head.return_twin = False
    base_model.model[-1] = twin_head
    
    # Load weights
    checkpoint = torch.load(weights_path, map_location='cpu', weights_only=False)
    state_dict = checkpoint.get('model', checkpoint)
    if hasattr(state_dict, 'state_dict'):
        state_dict = state_dict.state_dict()
    base_model.load_state_dict(state_dict, strict=False)
    base_model.to(device).eval()
    
    # Also load COCO model for full detection
    coco_model = YOLO("yolo11x.pt")
    coco_model.model.to(device).eval()
    
    return wrapper, coco_model


def draw_detections(image_path, twin_model, coco_model, conf_threshold=0.25):
    """
    Run detection and draw boxes.
    Green = Shuttle (class 0 from twin or class 80)
    Red = COCO classes
    """
    # Load image
    img = Image.open(image_path).convert('RGB')
    draw = ImageDraw.Draw(img)
    
    # Try to load a font, fall back to default
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 16)
    except:
        font = ImageFont.load_default()
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Run Twin model (for shuttle detection)
    twin_results = twin_model.predict(str(image_path), conf=conf_threshold, verbose=False, device=device)
    
    # Run COCO model (for other objects)
    coco_results = coco_model.predict(str(image_path), conf=conf_threshold, verbose=False, device=device)
    
    # Draw COCO detections (RED)
    if coco_results and len(coco_results) > 0 and coco_results[0].boxes:
        boxes = coco_results[0].boxes
        for i in range(len(boxes)):
            cls = int(boxes.cls[i].item())
            conf = boxes.conf[i].item()
            xyxy = boxes.xyxy[i].cpu().numpy()
            
            # Skip if it's likely a shuttle (we'll draw those in green)
            # Shuttle might be detected as class 0 (person) or other by COCO model
            # We'll let the Twin model handle shuttle
            
            # Draw red box for COCO
            x1, y1, x2, y2 = xyxy
            draw.rectangle([x1, y1, x2, y2], outline='red', width=3)
            
            # Label
            label = f"cls:{cls} {conf:.2f}"
            draw.text((x1, y1-20), label, fill='red', font=font)
    
    # Draw Shuttle detections (GREEN) - from Twin model
    if twin_results and len(twin_results) > 0 and twin_results[0].boxes:
        boxes = twin_results[0].boxes
        for i in range(len(boxes)):
            cls = int(boxes.cls[i].item())
            conf = boxes.conf[i].item()
            xyxy = boxes.xyxy[i].cpu().numpy()
            
            # Twin model outputs class 0 for shuttle
            if cls == 0 or cls == 80:
                x1, y1, x2, y2 = xyxy
                draw.rectangle([x1, y1, x2, y2], outline='lime', width=4)
                label = f"Shuttle {conf:.2f}"
                draw.text((x1, y1-22), label, fill='lime', font=font)
    
    return img


def create_grid(images, grid_size=5, cell_size=(400, 300)):
    """Create a grid of images."""
    grid_width = grid_size * cell_size[0]
    grid_height = grid_size * cell_size[1]
    
    grid = Image.new('RGB', (grid_width, grid_height), color='white')
    
    for idx, img in enumerate(images):
        if idx >= grid_size * grid_size:
            break
            
        # Resize image to cell size
        img_resized = img.resize(cell_size, Image.Resampling.LANCZOS)
        
        # Calculate position
        row = idx // grid_size
        col = idx % grid_size
        x = col * cell_size[0]
        y = row * cell_size[1]
        
        grid.paste(img_resized, (x, y))
    
    return grid


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', type=str, default='runs/detect/twin/weights/best.pt')
    parser.add_argument('--images-dir', type=str, default='data/test/images')
    parser.add_argument('--output', type=str, default='results/detection_grid.png')
    parser.add_argument('--grid-size', type=int, default=5)
    parser.add_argument('--conf', type=float, default=0.25)
    args = parser.parse_args()
    
    print(f"Loading Twin model from {args.weights}...")
    twin_model, coco_model = load_twin_model(args.weights)
    
    # Get image files
    images_dir = Path(args.images_dir)
    image_files = sorted(list(images_dir.glob("*.jpg")) + list(images_dir.glob("*.png")))
    
    # Select diverse images (spread across dataset)
    num_images = args.grid_size * args.grid_size
    total_images = len(image_files)
    
    if total_images <= num_images:
        selected_images = image_files
    else:
        # Sample evenly spaced images for diversity
        import random
        random.seed(42)  # For reproducibility
        
        # Take every N-th image to spread across dataset
        step = total_images // num_images
        indices = [i * step for i in range(num_images)]
        
        # Add some randomness within each segment
        selected_images = []
        for idx in indices:
            # Pick randomly within a window around the index
            window_start = max(0, idx - step//4)
            window_end = min(total_images - 1, idx + step//4)
            random_idx = random.randint(window_start, window_end)
            selected_images.append(image_files[random_idx])
    
    print(f"Selected {len(selected_images)} diverse images from {total_images} total...")
    
    processed_images = []
    for img_path in selected_images:
        print(f"  Processing {img_path.name}...")
        try:
            result_img = draw_detections(img_path, twin_model, coco_model, args.conf)
            processed_images.append(result_img)
        except Exception as e:
            print(f"    Error: {e}")
            # Use blank image as placeholder
            processed_images.append(Image.new('RGB', (640, 480), color='gray'))
    
    print(f"Creating {args.grid_size}x{args.grid_size} grid...")
    grid = create_grid(processed_images, grid_size=args.grid_size)
    
    # Save
    os.makedirs(os.path.dirname(args.output) if os.path.dirname(args.output) else '.', exist_ok=True)
    grid.save(args.output, quality=95)
    print(f"Saved grid to {args.output}")


if __name__ == "__main__":
    main()
