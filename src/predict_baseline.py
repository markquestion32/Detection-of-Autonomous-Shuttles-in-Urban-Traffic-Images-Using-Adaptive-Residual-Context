from ultralytics import YOLO
import sys
import os

def main():
    # 1. Paths
    # Adjust model path if your baseline run name differs
    model_path = "runs/detect/baseline_yolo11x/weights/best.pt"
    img_path = "bus.jpg"
    
    if not os.path.exists(model_path):
        print(f"Error: Model not found at {model_path}")
        # Try fallback or listing
        return
        
    if not os.path.exists(img_path):
        print(f"Error: Image not found at {img_path}")
        return

    # 2. Load Model
    print(f"Loading Baseline: {model_path}")
    model = YOLO(model_path)
    
    # 3. Predict
    print(f"Running Inference on {img_path}...")
    results = model.predict(img_path, conf=0.25, save=True, project="runs/predict", name="bus_test", exist_ok=True)
    
    # Ultralytics saves to runs/predict/bus_test/bus.jpg
    # Let's print the save dir
    print(f"\nResult saved to: {results[0].save_dir}")

if __name__ == "__main__":
    main()
