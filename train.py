import os
from ultralytics_mod.models.yolo import YOLO


os.environ["CUDA_VISIBLE_DEVICES"] = "0, 1"

project_save = "./runs/detect_1012"

dataset =  [
		".RTPE_Dataset.yaml",
            ]

model_config = ["./rts_yolov8/yolov8s.yaml",
                "./rts_yolov8/yolov5s.yaml",
                "./rts_yolov8/rts_yolov8s.yaml",
                "./rts_yolov8/RTS-YOLOv8s+PIoU.yaml",
                "./rts_yolov8/RTS-YOLOv8s+PIoU+P2.yaml",
                "./rts_yolov8/RTS-YOLOv8s+PIoU+P2+SPD.yaml",
                "./ultralytics_mod/cfg/models/v9/yolov9s.yaml",
                "./ultralytics_mod/cfg/models/v10/yolov10s.yaml",
                "./ultralytics_mod/cfg/models/v3/yolov3-tiny.yaml",
                ]

def train_yolov8():
    config = {
        "yaml_path": dataset[0],
        "model_name": model_config[2],

        "epochs": 300,
        "imgsz": 640,
        "batch": 8,
        "workers": 0,
        "device": "0",
        "project": project_save,
        "name": "RTS-YOLOv8s"
    }


    if not os.path.exists(config["yaml_path"]):
        raise FileNotFoundError(f"YAML config not exist: {config['yaml_path']}")

    model = YOLO(config["model_name"])
    # model.info(detailed=True)
    model.train(
        data=config["yaml_path"],
        imgsz=config["imgsz"],
        epochs=config["epochs"],
        batch=config["batch"],
        iou_type="Powerful-iou",
        workers=config["workers"],
        device=config["device"],
        project=config["project"],
        name=config["name"],
        exist_ok=True,
    )
    model_path = os.path.join(config["project"], config["name"], "weights", "best.pt")
    model.save(model_path)
    print(f"Model {config['name']} save to: {model_path}")

if __name__ == "__main__":
    train_yolov8()
