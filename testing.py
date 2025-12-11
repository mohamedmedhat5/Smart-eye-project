from ultralytics import YOLO
import os

model_path = 'project_results/coco_training/weights/best.pt'

def test_model_metrics():
    print("Calculating Model Metrics...")
    
    try:
        model = YOLO(model_path)
        metrics = model.val(data='coco8.yaml', split='val')
        
        print(f"mAP50: {metrics.box.map50:.2f}")
        print(f"Precision: {metrics.box.mp:.2f}")
        print(f"Recall: {metrics.box.mr:.2f}")
        
    except Exception as e:
        print(f"Error loading model: {e}")

def run_inference_demo():
    print("Running Inference on COCO Sample Images...")
    
    model = YOLO(model_path)
    
    # YOLOv8 automatically downloads coco8 samples to this path usually
    # We will predict on the validation set defined in the yaml
    
    results = model.predict(
        source='https://ultralytics.com/images/bus.jpg', 
        save=True,
        project='project_results',
        name='inference_demo',
        exist_ok=True,
        conf=0.25
    )
    print("Inference images saved at: project_results/inference_demo")

if __name__ == "__main__":
    test_model_metrics()
    run_inference_demo()