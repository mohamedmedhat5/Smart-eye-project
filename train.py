from ultralytics import YOLO

if __name__ == '__main__':
    model = YOLO('yolov8s.pt')

    results = model.train(
        data='coco8.yaml',
        epochs=50,
    
        imgsz=640,
        batch=16,
        workers=0,
        device=0,
        project='project_results',
        name='coco_training',
        exist_ok=True,
        plots=True
    )

    print("Training Completed. Results saved in project_results/coco_training")