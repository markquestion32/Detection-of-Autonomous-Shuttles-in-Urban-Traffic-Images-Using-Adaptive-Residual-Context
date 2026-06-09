import torch
import numpy as np
import torchvision
from ultralytics import YOLO
import sys
import os
import glob
import argparse
from PIL import Image, ImageDraw, ImageFont

# Import Architecture
sys.path.append(os.getcwd())
try:
    from src.twin_model import ContextGuidedDetect
except ImportError:
    pass

def xywh2xyxy(x):
    y = x.clone() if isinstance(x, torch.Tensor) else np.copy(x)
    y[..., 0] = x[..., 0] - x[..., 2] / 2
    y[..., 1] = x[..., 1] - x[..., 3] / 2
    y[..., 2] = x[..., 0] + x[..., 2] / 2
    y[..., 3] = x[..., 1] + x[..., 3] / 2
    return y

def run_nms(prediction, conf_thres=0.25, iou_thres=0.45):
    """ NMS for YOLO outputs """
    # DEBUG: Print max confidence
    if prediction is not None and len(prediction) > 0:
        # prediction is likely [1, 8400, 69] or similar after transpose? 
        # Actually standard YOLO is [1, 69, 8400]. 
        # run_nms handles the transpose.
        # Let's check max confidence in the raw prediction tensor before NMS
        pass

    if isinstance(prediction, (tuple, list)): prediction = prediction[0]
    if prediction is None: return []
    
    if prediction.ndim == 3 and prediction.shape[-1] > prediction.shape[-2]:
        prediction = prediction.transpose(-1, -2) 
    
    # Debug Max Conf
    if prediction.shape[1] > 4:
         max_conf = prediction[..., 4:].max().item()
         print(f"    [DEBUG] Max Raw Confidence in this head: {max_conf:.4f}")

    output = []
    for x in prediction:
        box = x[:, :4]
        cls_scores = x[:, 4:]
        conf, j = cls_scores.max(1, keepdim=True)
        mask = conf.view(-1) > conf_thres
        box = box[mask]
        conf = conf[mask]
        j = j[mask]
        if box.shape[0] == 0: 
            output.append(None)
            continue
        box = xywh2xyxy(box)
        keep = torchvision.ops.nms(box, conf.view(-1), iou_thres)
        det = torch.cat((box[keep], conf[keep], j[keep].float()), 1)
        output.append(det.cpu().numpy())
    return output


def letterbox_image(image, target_size=(640, 640)):
    """ 
    Resize image to target_size while preserving aspect ratio (letterboxing).
    Returns: new_image, (new_w, new_h), (pad_w, pad_h)
    """
    iw, ih = image.size
    w, h = target_size
    scale = min(w/iw, h/ih)
    nw = int(iw * scale)
    nh = int(ih * scale)

    image = image.resize((nw, nh), Image.BICUBIC)
    new_image = Image.new('RGB', target_size, (114, 114, 114)) # YOLO padding color
    
    # Paste centered
    pad_w = (w - nw) // 2
    pad_h = (h - nh) // 2
    new_image.paste(image, (pad_w, pad_h))
    
    return new_image, (nw, nh), (pad_w, pad_h)

def run_joint_inference(weights_path, source, conf_thres=0.15, output_dir="joint_results"):
    print(f">>> Starting Joint Inference (PIL) <<<")
    print(f"Weights: {weights_path}")
    print(f"Source: {source}")
    print(f"Confidence Threshold: {conf_thres}")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    os.makedirs(output_dir, exist_ok=True)

    # Load V3 Model
    print("Loading V3 Residual Model...")
    wrapper = YOLO("yolo11x.pt") 
    base_model = wrapper.model
    
    twin_head = ContextGuidedDetect(base_model.model[-1], nc_new=1)
    twin_head.return_twin = True # Enable twin output for joint inference
    base_model.model[-1] = twin_head
    
    print(f"Loading weights from {weights_path}...")
    try:
        checkpoint = torch.load(weights_path, map_location='cpu', weights_only=False)
    except Exception as e:
        print(f"CRITICAL ERROR: Failed to load weights: {e}")
        return

    state_dict = checkpoint['model'] if 'model' in checkpoint else checkpoint
    if hasattr(state_dict, 'state_dict'): 
        state_dict = state_dict.state_dict()
    

    # Strict load attempt
    base_model.load_state_dict(state_dict, strict=False)
    base_model.to(device).eval()
    
    # --- DUAL PASS SETUP ---
    print(">>> LOADING BASE MODEL FOR BACKGROUND (Dual Pass) <<<")
    fresh_wrapper = YOLO("yolo11x.pt") 
    fresh_wrapper.model.to(device).eval()

    coco_names = wrapper.model.names
    
    # Handle Source
    if os.path.isdir(source):
        img_files = glob.glob(os.path.join(source, "*"))
        img_files = [f for f in img_files if f.split('.')[-1].lower() in ['jpg', 'png', 'jpeg']]
    else:
        img_files = [source]
    
    if not img_files:
        print(f"No images found in {source}")
        return

    for img_path in img_files:
        print(f"Processing {os.path.basename(img_path)}...")
        try:
            img_pil = Image.open(img_path).convert("RGB")
        except Exception as e:
            print(f"Failed to load {img_path}: {e}")
            continue
            
        w0, h0 = img_pil.size
        
        # Preprocess with Letterbox
        img_input, (nw, nh), (pad_w, pad_h) = letterbox_image(img_pil, target_size=(640, 640))
        img_np = np.array(img_input)
        img_t = torch.from_numpy(img_np).to(device).float().permute(2,0,1).unsqueeze(0) / 255.0
        

        # --- DUAL PASS STRATEGY ---
        # Pass 1: Twin Model for Shuttle (Recruit)
        # Pass 2: Base YOLO Model for Background (Veteran functionality via separate pass)
        # This is necessary because the Twin Model's backbone drifted during training,
        # making feature maps incompatible with the original COCO head.
        
        # 1. Twin Model (Shuttles)
        with torch.no_grad():
            out = base_model(img_t)
            recruit_raw, _ = out # We ignore the drifting veteran head output
            
        shuttles = run_nms(recruit_raw, conf_thres=conf_thres)[0] 
        
        # 2. Fresh Base Model (Background)
        with torch.no_grad():
            # fresh_wrapper model expects standard YOLO input/processing
            # We can use the same img_t as it is standard resized/normalized
            bg_out = fresh_wrapper.model(img_t)
            # bg_out is usually just list of preds (or [preds]) depending on mode.
            # YOLO v8/11 predict() usually does postprocessing.
            # But calling .model() returns raw tensors.
            # Output of Detect head is list of 3 tensors (if export) or cat tensor (if standard).
            # Let's assume standard cat tensor [1, 84, 8400]
            
        # run_nms expects [1, 8400, 84] or similar
        # Check shape
        bg_preds = bg_out[0] if isinstance(bg_out, (tuple, list)) else bg_out
        
        background = run_nms(bg_preds, conf_thres=conf_thres)[0]
        
        if shuttles is None: shuttles = np.empty((0, 6))
        
        # Filter background to remove shuttles/buses/cars if they overlap twin shuttle? 
        # Or just keep everything not-shuttle.
        if background is None: background = np.empty((0, 6))

        # DEBUG PRINTS
        print(f"   - Found {len(shuttles)} Potential Shuttles (Twin), {len(background)} Background Objects (Base)")

        all_drawables = []
        
        # Scaling Factor for un-padding
        # x_original = (x_pad - pad_w) / scale
        scale = nw / w0 
        
        # 1. Shuttles
        for box in shuttles:
            b = box.copy()
            # Unpad then Unscale
            b[0] = (b[0] - pad_w) / scale
            b[1] = (b[1] - pad_h) / scale
            b[2] = (b[2] - pad_w) / scale
            b[3] = (b[3] - pad_h) / scale
            all_drawables.append({'box': b, 'type': 'shuttle'})
            
        # 2. Background (COCO)
        for box in background:
            b = box.copy()
            b[0] = (b[0] - pad_w) / scale
            b[1] = (b[1] - pad_h) / scale
            b[2] = (b[2] - pad_w) / scale
            b[3] = (b[3] - pad_h) / scale
            
            # VETO
            bad = False
            b_area = (b[2]-b[0])*(b[3]-b[1])
            for s in all_drawables:
                if s['type'] != 'shuttle': continue
                sbox = s['box']
                xx1 = max(b[0], sbox[0]); yy1 = max(b[1], sbox[1])
                xx2 = min(b[2], sbox[2]); yy2 = min(b[3], sbox[3])
                inter = max(0, xx2-xx1)*max(0, yy2-yy1)
                
                if inter/(b_area+1e-6) > 0.5:
                    bad = True
                    break
            if not bad:
                all_drawables.append({'box': b, 'type': 'coco'})

        # Draw
        draw = ImageDraw.Draw(img_pil)
        try:
            # Much larger font
            font = ImageFont.truetype("arial.ttf", 30) 
        except IOError:
            font = ImageFont.load_default()

        for item in all_drawables:
            box = item['box']
            x1, y1, x2, y2 = box[:4]
            conf = box[4]
            cls = int(box[5])
            
            if item['type'] == 'shuttle':
                label = f"Shuttle {conf:.2f}"
                color = "green"
                width = 5 # Thicker
            else:
                name = coco_names[cls] if cls < len(coco_names) else str(cls)
                label = f"{name} {conf:.2f}"
                color = "red"
                width = 3 # Thicker
            
            draw.rectangle([x1, y1, x2, y2], outline=color, width=width)
            
            try:
                text_bbox = draw.textbbox((x1, y1-30), label, font=font) # Move label up
                draw.rectangle(text_bbox, fill=color)
                draw.text((x1, y1-30), label, fill="white", font=font)
            except Exception:
                pass
            
        save_p = os.path.join(output_dir, f"joint_{os.path.basename(img_path)}")
        img_pil.save(save_p)
        print(f"Saved {save_p}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('source', type=str, help='Path to image or directory')
    parser.add_argument('--weights', type=str, default="runs/detect/twin_v3_RESIDUAL_yolo11x9/weights/best.pt", help='Path to trained weights')
    parser.add_argument('--conf', type=float, default=0.25, help='Confidence threshold')
    args = parser.parse_args()
    
    run_joint_inference(args.weights, args.source, conf_thres=args.conf)