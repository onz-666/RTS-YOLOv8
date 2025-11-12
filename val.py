from ultralytics_mod.models.yolo import YOLO
import os
import glob

image_dir = './rts_yolov8/RTSP/images/test'
output_dir = './val_folder/val_results'

models = ['./runs/detect_1012/RTS-YOLOv8/weights/best.pt', './runs/detect_1012/YOLOv8s/weights/best.pt']

os.makedirs(output_dir, exist_ok=True)

image_paths = glob.glob(os.path.join(image_dir, '*.jpg')) + \
              glob.glob(os.path.join(image_dir, '*.jpeg')) + \
              glob.glob(os.path.join(image_dir, '*.png'))

model_name=['RTS-YOLOv8', 'YOLOv8']
i = 0
for model_weight in models:
    model = YOLO(model_weight)
    
    save_path = output_dir + '/' + model_name[i]
    i = i + 1
    results = model.predict(image_dir, save=True, project=save_path, exist_ok=True)
        
        
    print(f"Results save to: {save_path}")