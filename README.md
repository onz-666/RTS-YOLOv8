# RTS-YOLOv8: Safety Operation Recognition in Railway Traction Power Supply System Based on Improved YOLOv8 Algorithm

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8.2](https://img.shields.io/badge/Python-3.8.2-blue.svg)](https://www.python.org/downloads/)
[![Ultralytics YOLOv8](https://img.shields.io/badge/Ultralytics-YOLOv8-red.svg)](https://github.com/ultralytics/ultralytics)

## Overview

Source code for RTS-YOLOv8, a YOLOv8-based model for real-time railway traction power supply system operation recognition (helmet, safebelt, safety clothes). Achieves mAP@0.5 0.949, 133 FPS on RTX 2080Ti.

## Dataset: Railway Traction Protective Equipment Dataset (RTPE)

- **Size**: 9600 images/videos from urban cameras (Helmet Wearing Detection Dataset[1], High-altitude operation and safety belt wearing dataset[2], SFCHD-SCALE Dataset[3]).
- **Classes**: 8 (no helmet, helmet, safebelt, offground, ground, red_armband, person, safety_clothes).
- **Splits**: Train (8,000 imgs), Val (800), Test (800). COCO annotations.
- **Access**: `./RTPE_DataSet.tar` (YAML config: `./rts_yolov8/RTPE_Dataset.yaml.yaml`).
- **License**: CC-BY 4.0.

Stats:
| Split | Images |
|-------|--------|
| Train | 8000 |
| Val   | 800   |
| Test  | 800   |

### Sources
The RTPE dataset is curated and annotated from the following public sources, with additional custom annotations for railway traction scenarios. All original datasets are cited and licensed appropriately.

- **[1] Safety Helmet Wearing Dataset** (njvisionpower, 2019): Images for helmet/no-helmet detection.  
  [GitHub Repository](https://github.com/njvisionpower/Safety-Helmet-Wearing-Dataset) (CC-BY-SA 4.0).

- **[2] High-Altitude Work and Safety Belt Wearing Dataset** (Baidu AI Studio, n.d.): Data for safebelt and offground/ground annotations.  
  [Baidu AI Studio](https://aistudio.baidu.com/datasetdetail/95663) (Open access, non-commercial use).

- **[3] Large, Complex, and Realistic Safety Clothing and Helmet Detection Dataset** (Yu et al., 2023): Base for safety_clothes, red_armband, and person classes.  
  [arXiv:2306.14975](https://arxiv.org/abs/2306.14975) (CC-BY 4.0).

For full details and licenses, see [LICENSE.dataset](LICENSE.dataset).

## Installation

1. Clone: `git clone https://github.com/onz-666/RTS-YOLOv8.git && cd RTS-YOLOv8`
2. Env: `python -m venv venv && source venv/bin/activate`
3. Deps: `pip install -r requirements.txt` (Ultralytics, PyTorch, tqdm).
4. GPU: Add CUDA via PyTorch index.

## Training and Validation

Use the `./train.py` script for training and validation. Edit the script to configure paths and hyperparameters before running.

### Quick Start
Run: `python train.py`

**Example Configuration** (`train.py` key parameters):
```python
import os
from ultralytics_mod.models.yolo import YOLO

os.environ["CUDA_VISIBLE_DEVICES"] = "0, 1"

project_save = "./runs/detect"  # Output directory
data = "./rts_yolov8/RTPE_Dataset.yaml"  # Dataset YAML
```

- **Output**: Weights and logs in `./runs/detect` (best.pt, results.csv).
- **Time**: ~40 hours on RTX 2080Ti for 300 epochs.
- **Metrics**: Precision, Recall, mAP in `./runs/detect/results.csv`. Validation auto-computed on val set.

## Inference (Prediction)

Use `./predict.py` for inference. Edit the script to configure paths before running.

### Quick Start
Run: `python predict.py`

**Example Configuration** (`predict.py` key parameters):
```python
from ultralytics_mod.models.yolo import YOLO
import os
import glob

image_dir = './RTPE_Dataset/images/test'  # Input directory
output_dir = './predict/results'  # Output directory
models = ['./runs/detect/weights/best.pt']  # Model list
```

- **Output**: Annotated images in `./val_folder/val_results/<model>/`; optional TXT labels.
- **Supports**: Images/videos/webcam; multi-model for ensemble.

## Results

The RTS-YOLOv8s model demonstrates superior performance on the RTPE dataset, with improvements in accuracy and robustness over baselines. Key metrics include mAP@0.5: 0.949 (vs. 0.934 for YOLOv8s) and inference speed of 8.1 ms on RTX 2080Ti.

### Performance Comparison of Detection Models

| Models          | P     | R     | mAP@0.5 | mAP@0.5:0.95 | Params/M | FLOPs/G | Inference Speed /ms |
|-----------------|-------|-------|---------|--------------|----------|---------|---------------------|
| YOLOv3-tiny    | 0.921 | 0.760 | 0.830  | 0.585       | 12.13   | 18.9   | 3.5                |
| YOLOv5s        | 0.930 | 0.879 | 0.928  | 0.675       | 9.13    | 24.1   | 4.6                |
| YOLOv8s        | 0.936 | 0.882 | 0.933  | 0.688       | 11.13   | 28.5   | 5.6                |
| YOLOv9s        | 0.925 | 0.890 | 0.936  | 0.693       | 7.17    | 26.7   | 6.0                |
| YOLOv10s       | 0.921 | 0.869 | 0.925  | 0.682       | 8.04    | 24.5   | 3.4                |
| **RTS-YOLOv8s**| **0.939** | **0.896** | **0.949** | **0.708** | **9.94** | **32.8** | **8.1** |

### Comparison of Detection Results on the RTPE Dataset

| Categories     | YOLOv8s (mAP@0.5) | RTS-YOLOv8s (mAP@0.5) |
|----------------|-------------------|-----------------------|
| **all**       | 0.934            | **0.949**            |
| no helmet     | 0.928            | **0.940**            |
| helmet        | 0.901            | **0.926**            |
| safebelt      | 0.916            | **0.936**            |
| offground     | 0.977            | 0.975                |
| ground        | 0.951            | **0.958**            |
| red armband   | 0.894            | **0.937**            |
| person        | 0.957            | **0.966**            |
| safety clothes| 0.944            | **0.954**            |

## Citation

```
@article{hao2025rts,
  title = {RTS-YOLOv8: Safety Operation Recognition in Railway Traction Power Supply System Based on Improved YOLOv8 Algorithm},
  author = {Hao, Guicai and Lei, Zhe and Lu, Jian and Guo, Xinquan and Pei, Yalin},
  journal = {},
  volume = {},
  number = {},
  pages = {},
  year = {},
  publisher = {},
  doi = {},
  url = {}
}
```

## License & Contact

MIT License. Contact: lei06z@163.com. Issues: GitHub.

*Updated: Nov 2025*
