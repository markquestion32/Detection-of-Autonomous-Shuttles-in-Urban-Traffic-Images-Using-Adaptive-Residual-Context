"""
Clean Evaluation Script - Uses Ultralytics built-in validation for accurate metrics.
Evaluates all experiments and outputs mAP@50 and mAP@50:95.
"""
import os
import sys
import json
import torch
from pathlib import Path
from ultralytics import YOLO

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))


def evaluate_shuttle(model, data_yaml, split='test'):
    """Evaluate on shuttle dataset using Ultralytics val()."""
    print(f"  Running Ultralytics validation on shuttle ({split})...")
    
    try:
        metrics = model.val(
            data=str(data_yaml),
            split=split,
            conf=0.001,  # Low conf for accurate mAP
            iou=0.6,
            verbose=False,
            plots=False
        )
        
        return {
            'map50': float(metrics.box.map50),
            'map5095': float(metrics.box.map),
            'precision': float(metrics.box.mp),
            'recall': float(metrics.box.mr)
        }
    except Exception as e:
        print(f"  ERROR: {e}")
        return {'map50': 0.0, 'map5095': 0.0, 'precision': 0.0, 'recall': 0.0}


def evaluate_coco(model, coco_yaml, split='val'):
    """Evaluate on COCO using Ultralytics val()."""
    print(f"  Running Ultralytics validation on COCO ({split})...")
    
    try:
        metrics = model.val(
            data=str(coco_yaml),
            split=split,
            conf=0.001,
            iou=0.6,
            verbose=False,
            plots=False
        )
        
        return {
            'map50': float(metrics.box.map50),
            'map5095': float(metrics.box.map),
            'precision': float(metrics.box.mp),
            'recall': float(metrics.box.mr)
        }
    except Exception as e:
        print(f"  ERROR: {e}")
        return {'map50': 0.0, 'map5095': 0.0, 'precision': 0.0, 'recall': 0.0}


def load_twin_model(weights_path, device):
    """Load Twin model architecture."""
    from src.twin_model import ContextGuidedDetect
    
    wrapper = YOLO("yolo11x.pt")
    base_model = wrapper.model
    
    twin_head = ContextGuidedDetect(base_model.model[-1], nc_new=1)
    base_model.model[-1] = twin_head
    
    checkpoint = torch.load(weights_path, map_location='cpu', weights_only=False)
    state_dict = checkpoint.get('model', checkpoint)
    if hasattr(state_dict, 'state_dict'):
        state_dict = state_dict.state_dict()
    base_model.load_state_dict(state_dict, strict=False)
    base_model.to(device).eval()
    
    return wrapper


def main():
    project_root = Path(__file__).parent.parent
    results_dir = project_root / "results"
    results_dir.mkdir(exist_ok=True)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Data configs
    shuttle_yaml = project_root / "data" / "data.yaml"  # Original shuttle data
    shuttle_81_yaml = project_root / "data" / "data_81class.yaml"  # 81-class shuttle
    coco_yaml = project_root / "data" / "coco.yaml"  # COCO validation
    
    # Define experiments
    experiments = [
        {
            'name': 'E0_Pretrained',
            'weights': 'yolo11x.pt',
            'type': 'pretrained',
            'shuttle_yaml': None,  # Can't evaluate shuttle (no class 80)
            'coco_yaml': coco_yaml
        },
        {
            'name': 'E1_UpperBound',
            'weights': project_root / 'runs/detect/e1_upperbound/weights/best.pt',
            'type': 'finetuned_81',
            'shuttle_yaml': shuttle_81_yaml,
            'coco_yaml': coco_yaml
        },
        {
            'name': 'E2_Naive',
            'weights': project_root / 'runs/detect/e2_naive/weights/best.pt',
            'type': 'finetuned_81',
            'shuttle_yaml': shuttle_81_yaml,
            'coco_yaml': coco_yaml
        },
        {
            'name': 'E3_LwF',
            'weights': project_root / 'runs/detect/e3_lwf/weights/best.pt',
            'type': 'finetuned_81',
            'shuttle_yaml': shuttle_81_yaml,
            'coco_yaml': coco_yaml
        },
        {
            'name': 'E4_Twin',
            'weights': project_root / 'runs/detect/twin/weights/best.pt',
            'type': 'twin',
            'shuttle_yaml': shuttle_yaml,  # 1-class for twin
            'coco_yaml': coco_yaml
        },
        {
            'name': 'E5_Transfer',
            'weights': project_root / 'runs/detect/e5_transfer/weights/best.pt',
            'type': 'finetuned_81',
            'shuttle_yaml': shuttle_81_yaml,
            'coco_yaml': coco_yaml
        },
        {
            'name': 'E6_ShuttleOnly',
            'weights': project_root / 'runs/detect/e6_shuttle_only/weights/best.pt',
            'type': 'finetuned_1',
            'shuttle_yaml': shuttle_yaml,  # 1-class
            'coco_yaml': None  # Can't eval COCO (only 1 class)
        },
    ]
    
    all_results = []
    
    print("=" * 80)
    print("CLEAN EVALUATION - Using Ultralytics Built-in Validation")
    print("=" * 80)
    
    for exp in experiments:
        print(f"\n{'='*60}")
        print(f"Evaluating: {exp['name']}")
        print(f"{'='*60}")
        
        # Check weights exist
        weights_path = exp['weights']
        if isinstance(weights_path, Path) and not weights_path.exists():
            print(f"  SKIPPING: Weights not found at {weights_path}")
            continue
        
        # Load model
        print(f"  Loading model: {weights_path}")
        
        if exp['type'] == 'twin':
            model = load_twin_model(str(weights_path), device)
            coco_model = YOLO("yolo11x.pt")  # Separate COCO model for twin
        else:
            model = YOLO(str(weights_path))
            coco_model = None
        
        result = {'name': exp['name']}
        
        # Evaluate Shuttle
        if exp['shuttle_yaml'] is not None:
            print(f"\n  [Shuttle Evaluation]")
            shuttle_metrics = evaluate_shuttle(model, exp['shuttle_yaml'])
            result['shuttle_map50'] = shuttle_metrics['map50'] * 100
            result['shuttle_map5095'] = shuttle_metrics['map5095'] * 100
            print(f"    mAP@50: {result['shuttle_map50']:.1f}%")
            print(f"    mAP@50:95: {result['shuttle_map5095']:.1f}%")
        else:
            result['shuttle_map50'] = None
            result['shuttle_map5095'] = None
            print(f"\n  [Shuttle Evaluation] SKIPPED (no shuttle class)")
        
        # Evaluate COCO
        if exp['coco_yaml'] is not None:
            print(f"\n  [COCO Evaluation]")
            
            if exp['type'] == 'twin':
                # For twin, use the separate COCO model
                coco_metrics = evaluate_coco(coco_model, exp['coco_yaml'])
            else:
                coco_metrics = evaluate_coco(model, exp['coco_yaml'])
            
            result['coco_map50'] = coco_metrics['map50'] * 100
            result['coco_map5095'] = coco_metrics['map5095'] * 100
            print(f"    mAP@50: {result['coco_map50']:.1f}%")
            print(f"    mAP@50:95: {result['coco_map5095']:.1f}%")
        else:
            result['coco_map50'] = None
            result['coco_map5095'] = None
            print(f"\n  [COCO Evaluation] SKIPPED (no COCO classes)")
        
        all_results.append(result)
    
    # Print summary table
    print("\n" + "=" * 80)
    print("RESULTS SUMMARY")
    print("=" * 80)
    print(f"{'Experiment':<20} | {'Shuttle mAP@50':>14} | {'Shuttle mAP@50:95':>17} | {'COCO mAP@50':>12} | {'COCO mAP@50:95':>15}")
    print("-" * 90)
    
    for r in all_results:
        shuttle_50 = f"{r['shuttle_map50']:.1f}%" if r['shuttle_map50'] is not None else "N/A"
        shuttle_95 = f"{r['shuttle_map5095']:.1f}%" if r['shuttle_map5095'] is not None else "N/A"
        coco_50 = f"{r['coco_map50']:.1f}%" if r['coco_map50'] is not None else "N/A"
        coco_95 = f"{r['coco_map5095']:.1f}%" if r['coco_map5095'] is not None else "N/A"
        print(f"{r['name']:<20} | {shuttle_50:>14} | {shuttle_95:>17} | {coco_50:>12} | {coco_95:>15}")
    
    # Save results
    output_file = results_dir / "clean_evaluation.json"
    with open(output_file, 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults saved to: {output_file}")
    
    # Save as CSV
    csv_file = results_dir / "clean_evaluation.csv"
    with open(csv_file, 'w') as f:
        f.write("Experiment,Shuttle_mAP50,Shuttle_mAP5095,COCO_mAP50,COCO_mAP5095\n")
        for r in all_results:
            shuttle_50 = f"{r['shuttle_map50']:.1f}" if r['shuttle_map50'] is not None else ""
            shuttle_95 = f"{r['shuttle_map5095']:.1f}" if r['shuttle_map5095'] is not None else ""
            coco_50 = f"{r['coco_map50']:.1f}" if r['coco_map50'] is not None else ""
            coco_95 = f"{r['coco_map5095']:.1f}" if r['coco_map5095'] is not None else ""
            f.write(f"{r['name']},{shuttle_50},{shuttle_95},{coco_50},{coco_95}\n")
    print(f"CSV saved to: {csv_file}")


if __name__ == "__main__":
    main()
