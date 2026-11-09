## 📁 Project Structure

```text
yolov7-from-scratch/
├── models/
│   ├── __init__.py
│   ├── repconv.py
│   ├── elan.py
│   ├── head.py
│   └── yolo.py
│
├── utils/
│   ├── __init__.py
│   ├── loss.py
│   ├── assigner.py
│   ├── metrics.py
│   ├── boxes.py
│   └── general.py
│
├── data/
│   ├── __init__.py
│   ├── dataset.py
│   └── augmentations.py
│
├── tests/
│   ├── test_repconv.py
│   └── test_forward.py
│
├── configs/
│   └── yolov7.yaml
│
├── train.py
├── val.py
├── detect.py
└── README.md
```