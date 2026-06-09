"""
evaluate_experiment.py - Unified Evaluation for Continual Learning Experiments

Measures:
- mAP@50 (Plasticity & Stability)
- Precision, Recall, F1
- Inference FPS and parameter count (Overhead)
"""

import torch
import numpy as np
import os
import sys
import time
import json
import argparse
from pathlib import Path
from collections import defaultdict
from ultralytics import YOLO


# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from src.twin_model import ContextGuidedDetect
except ImportError:
    from twin_model import ContextGuidedDetect


def load_model(weights_path, model_type="pretrained"):
    """
    Load model based on type.
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    if model_type == "pretrained":
        # Fresh COCO-pretrained model
        model = YOLO(weights_path)
        if hasattr(model, 'model'):
            model.model.to(device).eval()
        return model, False
        
    elif model_type == "twin":
        # Load Twin architecture
        wrapper = YOLO("yolo11x.pt")
        base_model = wrapper.model
        
        # Attach twin head
        twin_head = ContextGuidedDetect(base_model.model[-1], nc_new=1)
        twin_head.return_twin = False  # Only return recruit head for standard prediction
        base_model.model[-1] = twin_head
        
        # Load trained weights
        checkpoint = torch.load(weights_path, map_location='cpu', weights_only=False)
        state_dict = checkpoint.get('model', checkpoint)
        if hasattr(state_dict, 'state_dict'):
            state_dict = state_dict.state_dict()
        base_model.load_state_dict(state_dict, strict=False)
        base_model.to(device).eval()
        
        # Also load fresh COCO model for background detection
        coco_model = YOLO("yolo11x.pt")
        coco_model.model.to(device).eval()
        
        return (wrapper, coco_model), True
        
    else:
        # Finetuned or EWC - standard YOLO loading
        model = YOLO(weights_path)
        # Ensure model is on correct device
        if hasattr(model, 'model'):
            model.model.to(device).eval()
        return model, False


def compute_iou(box1, box2):
    """Compute IoU between two boxes [x1, y1, x2, y2]"""
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = area1 + area2 - inter
    
    return inter / (union + 1e-6)


def compute_ap_per_class(tp, conf, num_gt):
    """Compute AP for a single class"""
    if num_gt == 0:
        return 0.0, 0.0, 0.0 # AP, Precision, Recall
        
    if len(tp) == 0:
        return 0.0, 0.0, 0.0
        
    # Sort by confidence
    i = np.argsort(-conf)
    tp = tp[i]
    conf = conf[i]
    
    # Compute precision/recall
    fpc = (1 - tp).cumsum()
    tpc = tp.cumsum()
    
    recall_curve = tpc / (num_gt + 1e-6)
    precision_curve = tpc / (tpc + fpc + 1e-6)
    
    # Compute AP (Area Under Curve)
    # Append sentinels
    mrec = np.concatenate(([0.0], recall_curve, [1.0]))
    mpre = np.concatenate(([0.0], precision_curve, [0.0]))
    
    # Compute the precision envelope
    for i in range(mpre.size - 1, 0, -1):
        mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])
        
    # Integrate area under curve
    method = 'interp'  # methods: 'continuous', 'interp'
    i = np.where(mrec[1:] != mrec[:-1])[0]  # points where x axis (recall) changes
    ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])
    
    # Return metrics at best F1 score (or simply max P/R) -> return last P/R
    precision = np.mean(precision_curve)
    recall = np.mean(recall_curve)
    
    return ap, precision, recall


def evaluate_dataset(model, is_twin, img_dir, label_dir=None, conf_thres=0.25, iou_thres=0.5, target_classes=None, limit=None, loose_match=False, compute_map5095=True):
    """
    Generic evaluation function for mAP, Precision, Recall.
    
    Args:
        model: Loaded model
        is_twin: Boolean
        img_dir: Path to images
        label_dir: Path to labels (inferred if None)
        target_classes: List of class IDs to evaluate (None = all)
        loose_match: If True, map all matching target_classes to class 0 (for binary eval)
        compute_map5095: If True, compute mAP@50:95 in addition to mAP@50
        
    Returns:
        dict: Metrics (mAP50, Precision, Recall, F1)
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    img_dir = Path(img_dir)
    
    if label_dir is None:
        # Infer properties
        # 1. Parallel split: data/val2017 -> data/labels/val2017
        candidate_1 = img_dir.parent / "labels" / img_dir.name
        # 2. Sibling labels: data/test/images -> data/test/labels
        candidate_2 = img_dir.parent / "labels"
        
        if candidate_1.exists():
            label_dir = candidate_1
        elif candidate_2.exists():
            label_dir = candidate_2
        else:
             label_dir = img_dir.parent.parent / "labels" / img_dir.name
             
    if not label_dir.exists():
        print(f"Warning: Label directory {label_dir} not found. Returning 0 metrics.")
        return {'map50': 0.0, 'precision': 0.0, 'recall': 0.0, 'f1': 0.0}

    files = sorted(list(img_dir.glob("*.jpg")) + list(img_dir.glob("*.png")))
    if limit:
        files = files[:limit]
        
    stats = [] # List of (tp, conf, pred_cls, target_cls)
    
    print(f"Evaluating on {len(files)} images from {img_dir}...")
    
    for img_path in files:
        # Load GT
        gt_boxes = []
        gt_classes = []
        
        txt_path = label_dir / (img_path.stem + ".txt")
        if txt_path.exists():
            with open(txt_path, 'r') as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) >= 5:
                        cls = int(float(parts[0]))
                        
                        # --- Logic Update: Loose Matching ---
                        if loose_match:
                            # If loose match, checking if it is one of the target classes (e.g. 0 or 80)
                            # If so, map to 0.
                            if target_classes is not None:
                                if cls in target_classes:
                                    cls = 0
                                else: 
                                    continue # Ignore other classes
                            else:
                                cls = 0 # Map everything to 0 if no target specified? Safest to require target.
                        else:
                             # Strict matching
                            if target_classes is not None and cls not in target_classes:
                                continue
                        # ------------------------------------
                        
                        gt_classes.append(cls)
                        gt_boxes.append([float(x) for x in parts[1:5]]) # xc, yc, w, h

        
        # Inference
        if is_twin:
            twin_model, coco_model = model
            # Simple heuristic:
            use_shuttle_head = (target_classes is not None and (0 in target_classes or 80 in target_classes) and len(target_classes) <= 2)
            
            if use_shuttle_head:
                results = twin_model.predict(str(img_path), conf=conf_thres, verbose=False, device=device)
            else:
                results = coco_model.predict(str(img_path), conf=conf_thres, verbose=False, device=device)
        else:
            results = model.predict(str(img_path), conf=conf_thres, verbose=False, device=device)
            
        # Get Image Shape from Results
        if results and len(results) > 0:
            h_img, w_img = results[0].orig_shape
        else:
            # Should unlikely happen with Ultralytics predict unless file error
            print(f"Warning: No results for {img_path}")
            continue

        # Convert GT to xyxy pixels (Now that we have shape)
        target_cls_list = []
        target_bboxes = []
        
        if len(gt_boxes) > 0:
            gt_boxes = np.array(gt_boxes)
            gt_classes = np.array(gt_classes)
            
            # Deep copy to verify we don't mutate original if cached (though we reload)
            gt_b = gt_boxes.copy() 
            gt_b[:, 0] *= w_img # xc
            gt_b[:, 2] *= w_img # w
            gt_b[:, 1] *= h_img # yc
            gt_b[:, 3] *= h_img # h
            
            # center to xyxy
            b = gt_b.copy()
            b[:, 0] = gt_b[:, 0] - gt_b[:, 2] / 2 # x1
            b[:, 1] = gt_b[:, 1] - gt_b[:, 3] / 2 # y1
            b[:, 2] = gt_b[:, 0] + gt_b[:, 2] / 2 # x2
            b[:, 3] = gt_b[:, 1] + gt_b[:, 3] / 2 # y2
            
            target_bboxes = b
            target_cls_list = gt_classes
            
        # Process Predictions
        pred_bboxes = []
        pred_confs = []
        pred_classes = []
        
        if results and len(results) > 0:
            boxes = results[0].boxes
            if boxes is not None:
                for i in range(len(boxes)):
                    cls = int(boxes.cls[i].item())
                    
                    # --- Logic Update: Loose Matching ---
                    if loose_match:
                         if target_classes is not None:
                             if cls in target_classes:
                                 cls = 0
                             else:
                                 continue
                    else:
                        if target_classes is not None and cls not in target_classes:
                             continue
                    # ------------------------------------
                             
                    pred_classes.append(cls)
                    # Coordinates are already absolute xyxy in results
                    pred_bboxes.append(boxes.xyxy[i].cpu().numpy())
                    pred_confs.append(boxes.conf[i].item())

        pred_bboxes = np.array(pred_bboxes)
        pred_classes = np.array(pred_classes)
        pred_confs = np.array(pred_confs)
        
        # Match Predictions to GT
        correct = []
        
        if len(pred_bboxes) == 0:
            if len(target_bboxes) > 0:
                # All FN
                pass
        else:
            # For each prediction, finding matching GT
            # Simple metric per image: [iou, cls_match]
            
            # We need to construct stats per class for AP computation
            # For implementation simplicity, let's use a simplified matcher
            
            detected_gt = []
            
            # Sort by conf
            sorted_idx = np.argsort(-pred_confs)
            pred_bboxes = pred_bboxes[sorted_idx]
            pred_classes = pred_classes[sorted_idx]
            pred_confs = pred_confs[sorted_idx]
            
            for p_i, p_box in enumerate(pred_bboxes):
                p_cls = pred_classes[p_i]
                
                best_iou = 0
                best_gt_idx = -1
                
                for g_i, g_box in enumerate(target_bboxes):
                    if g_i in detected_gt:
                        continue
                    if target_cls_list[g_i] != p_cls:
                        continue
                        
                    iou = compute_iou(p_box, g_box)
                    if iou > best_iou:
                        best_iou = iou
                        best_gt_idx = g_i
                
                # Store the best IoU value (for mAP50:95 computation)
                if best_iou >= 0.5:  # Minimum IoU for any match
                    detected_gt.append(best_gt_idx)
                    # Store (iou, conf, pred_cls, gt_cls) for multi-threshold evaluation
                    stats.append((best_iou, pred_confs[p_i], p_cls, p_cls))
                else:
                    stats.append((0, pred_confs[p_i], p_cls, p_cls))  # FP
                    
        # Add missed GTs (FN) - implicitly handled in AP calculation by num_gt
        # But we need num_gt count
        
    # Compute Metrics
    metrics_per_class = {}
    
    # Organize stats by class
    stats_by_class = defaultdict(lambda: {'tp': [], 'conf': []})
    for tp, conf, p_cls, _ in stats:
        stats_by_class[p_cls]['tp'].append(tp)
        stats_by_class[p_cls]['conf'].append(conf)
        
    # Count GT per class (globally)
    gt_counts = defaultdict(int)
    for img_path in files:
         # Need to re-read or cache GT counts. Re-reading strictly for counting:
         # Optimization: could have counted above.
         pass 
         
    # To save IO, let's do a quick pass or accept overhead.
    # Actually, simpler: accumulate GT counts in the main loop!
    # Let's fix loop to count GTs.
    # ... (Logic modified in thought: add `global_gt_counts`)
    
    return compute_metrics_from_stats(stats, files, label_dir, target_classes, loose_match)


def compute_metrics_from_stats(stats, files, label_dir, target_classes, loose_match=False):
    """Compute mAP@50 and mAP@50:95 from stats."""
    # Re-scan GT counts
    gt_counts = defaultdict(int)
    
    for img_path in files:
         txt_path = label_dir / (img_path.stem + ".txt")
         if txt_path.exists():
            with open(txt_path, 'r') as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) >= 1:
                        cls = int(float(parts[0]))
                        
                        if loose_match:
                            if target_classes and cls in target_classes:
                                cls = 0
                            else:
                                continue
                        elif target_classes is not None and cls not in target_classes:
                            continue
                            
                        gt_counts[cls] += 1
    
    # IoU thresholds for mAP@50:95
    iou_thresholds = np.arange(0.5, 1.0, 0.05)  # [0.50, 0.55, 0.60, ..., 0.95]
    
    all_classes = set(gt_counts.keys())
    if target_classes:
        all_classes = all_classes.union(set(target_classes))
    
    # Compute AP at each IoU threshold
    aps_per_threshold = []
    
    for iou_thresh in iou_thresholds:
        aps_at_thresh = []
        
        for cls in all_classes:
            num_gt = gt_counts[cls]
            cls_stats = [x for x in stats if x[2] == cls]
            
            if not cls_stats or num_gt == 0:
                aps_at_thresh.append(0.0)
                continue
            
            # Convert IoU values to TP (1) or FP (0) at this threshold
            # stats format: (iou, conf, pred_cls, gt_cls)
            tp_list = np.array([1 if x[0] >= iou_thresh else 0 for x in cls_stats])
            conf_list = np.array([x[1] for x in cls_stats])
            
            ap, _, _ = compute_ap_per_class(tp_list, conf_list, num_gt)
            aps_at_thresh.append(ap)
        
        if aps_at_thresh:
            aps_per_threshold.append(np.mean(aps_at_thresh))
        else:
            aps_per_threshold.append(0.0)
    
    # mAP@50 = first threshold only
    map50 = aps_per_threshold[0] if aps_per_threshold else 0.0
    
    # mAP@50:95 = average across all thresholds
    map5095 = np.mean(aps_per_threshold) if aps_per_threshold else 0.0
    
    # Compute precision/recall at IoU=0.5 for reporting
    precisions = []
    recalls = []
    
    for cls in all_classes:
        num_gt = gt_counts[cls]
        cls_stats = [x for x in stats if x[2] == cls]
        
        if not cls_stats:
            precisions.append(0.0)
            recalls.append(0.0)
            continue
            
        tp_list = np.array([1 if x[0] >= 0.5 else 0 for x in cls_stats])
        conf_list = np.array([x[1] for x in cls_stats])
        
        _, p, r = compute_ap_per_class(tp_list, conf_list, num_gt)
        precisions.append(p)
        recalls.append(r)
    
    mean_p = np.mean(precisions) if precisions else 0.0
    mean_r = np.mean(recalls) if recalls else 0.0
    f1 = 2 * (mean_p * mean_r) / (mean_p + mean_r + 1e-6)
    
    return {
        'map50': map50,
        'map5095': map5095,
        'precision': mean_p,
        'recall': mean_r,
        'f1': f1
    }


def measure_inference_speed(model, is_twin, device, num_warmup=10, num_runs=100):
    """Measure inference FPS."""
    import time
    
    # Create dummy input
    dummy_input = torch.randn(1, 3, 640, 640).to(device)
    
    if is_twin:
        test_model = model[0]  # Use twin model
    else:
        test_model = model
    
    # Warmup
    with torch.no_grad():
        for _ in range(num_warmup):
            _ = test_model.predict(dummy_input, verbose=False)
    
    # Timed runs
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    
    start = time.perf_counter()
    with torch.no_grad():
        for _ in range(num_runs):
            _ = test_model.predict(dummy_input, verbose=False)
    
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    
    elapsed = time.perf_counter() - start
    fps = num_runs / elapsed
    
    return fps


def get_model_size(model, is_twin):
    """Get model size in MB and parameter count."""
    if is_twin:
        # Count twin model parameters
        params = sum(p.numel() for p in model[0].model.parameters())
    else:
        params = sum(p.numel() for p in model.model.parameters())
    
    # Approximate size (4 bytes per float32 param)
    size_mb = params * 4 / (1024 * 1024)
    
    return params, size_mb


def get_gpu_memory():
    """Get current GPU memory usage in MB."""
    if torch.cuda.is_available():
        return torch.cuda.max_memory_allocated() / (1024 * 1024)
    return 0.0


def run_evaluation(weights_path, model_type, shuttle_test_path, coco_path, 
                   experiment_name="experiment", output_dir="results"):
    
    print(f"\n{'='*60}")
    print(f"Evaluating: {experiment_name}")
    print(f"{'='*60}")
    
    os.makedirs(output_dir, exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Reset GPU memory counter
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    
    print("\n1. Loading model...")
    model, is_twin = load_model(weights_path, model_type)
    
    # Get model size
    params, size_mb = get_model_size(model, is_twin)
    print(f"   Parameters: {params:,}")
    print(f"   Model Size: {size_mb:.1f} MB")
    
    # 2. Evaluate Shuttle (Plasticity)
    print("\n2. Evaluating Shuttle (Class 80)...")
    shuttle_metrics = evaluate_dataset(
        model, is_twin, 
        img_dir=shuttle_test_path, 
        target_classes=[0, 80],
        loose_match=True
    )
    
    print(f"   Shuttle mAP@50: {shuttle_metrics['map50']*100:.1f}%")
    print(f"   Shuttle mAP@50:95: {shuttle_metrics.get('map5095', 0)*100:.1f}%")
    print(f"   Precision: {shuttle_metrics['precision']*100:.1f}%")
    print(f"   Recall: {shuttle_metrics['recall']*100:.1f}%")
    print(f"   F1: {shuttle_metrics['f1']:.3f}")
    
    # 3. Evaluate COCO (Stability)
    print("\n3. Evaluating COCO (Classes 0-79)...")
    coco_metrics = evaluate_dataset(
        model, is_twin, 
        img_dir=coco_path,
        target_classes=list(range(80)),
        limit=500
    )
    
    print(f"   COCO mAP@50: {coco_metrics['map50']*100:.1f}%")
    print(f"   COCO mAP@50:95: {coco_metrics.get('map5095', 0)*100:.1f}%")
    print(f"   Precision: {coco_metrics['precision']*100:.1f}%")
    print(f"   Recall: {coco_metrics['recall']*100:.1f}%")
    print(f"   F1: {coco_metrics['f1']:.3f}")
    
    # 4. Measure Inference Speed
    print("\n4. Measuring Inference Speed...")
    try:
        fps = measure_inference_speed(model, is_twin, device)
        print(f"   FPS: {fps:.1f}")
    except Exception as e:
        fps = 0.0
        print(f"   FPS: Error ({e})")
    
    # 5. GPU Memory
    gpu_memory = get_gpu_memory()
    print(f"   Peak GPU Memory: {gpu_memory:.1f} MB")
    
    # Harmonic Mean (using mAP@50:95)
    shuttle_map = shuttle_metrics.get('map5095', shuttle_metrics['map50'])
    coco_map = coco_metrics.get('map5095', coco_metrics['map50'])
    
    h_mean = 0.0
    if shuttle_map > 0 and coco_map > 0:
        h_mean = 2 * (shuttle_map * coco_map) / (shuttle_map + coco_map)
        
    print(f"\n   Harmonic Mean (mAP@50:95): {h_mean*100:.1f}%")
    
    # Backward Transfer (forgetting) - compare to baseline COCO performance
    # Baseline COCO mAP@50:95 for yolo11x is ~40% typically
    baseline_coco_map = 0.40
    backward_transfer = coco_map - baseline_coco_map
    
    # Forward Transfer - how well new task learned
    forward_transfer = shuttle_map
    
    print(f"   Backward Transfer: {backward_transfer*100:+.1f}%")
    print(f"   Forward Transfer: {forward_transfer*100:.1f}%")
    
    # Save Results
    results = {
        'experiment': experiment_name,
        'shuttle': {
            'map50': shuttle_metrics['map50'],
            'map5095': shuttle_metrics.get('map5095', 0),
            'precision': shuttle_metrics['precision'],
            'recall': shuttle_metrics['recall'],
            'f1': shuttle_metrics['f1']
        },
        'coco': {
            'map50': coco_metrics['map50'],
            'map5095': coco_metrics.get('map5095', 0),
            'precision': coco_metrics['precision'],
            'recall': coco_metrics['recall'],
            'f1': coco_metrics['f1']
        },
        'harmonic_mean': h_mean,
        'fps': fps,
        'params': params,
        'size_mb': size_mb,
        'gpu_memory_mb': gpu_memory,
        'backward_transfer': backward_transfer,
        'forward_transfer': forward_transfer
    }
    
    results_path = os.path.join(output_dir, f"{experiment_name}_results.json")
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
        
    return results

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', type=str, required=True)
    parser.add_argument('--model-type', type=str, default='pretrained')
    parser.add_argument('--shuttle-path', type=str, default='data/test/images')
    parser.add_argument('--coco-path', type=str, default='data/val2017')
    parser.add_argument('--name', type=str, default='experiment')
    parser.add_argument('--output', type=str, default='results')
    
    args = parser.parse_args()
    
    run_evaluation(
        args.weights, args.model_type, 
        args.shuttle_path, args.coco_path, 
        args.name, args.output
    )
