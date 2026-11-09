yolov7-from-scratch/
│
├── models/
│   ├── __init__.py
│   ├── repconv.py        # RepConv layer & fusion logic
│   ├── elan.py           # E-ELAN & downsampling modules (MPConv)
│   ├── head.py           # Lead & Aux detection heads
│   └── yolo.py           # Full architecture assembly
│
├── utils/
│   ├── __init__.py
│   ├── loss.py           # CIoU, BCEWithLogits compound loss
│   ├── assigner.py       # SimOTA / Lead-guided dynamic assigner
│   ├── metrics.py        # mAP calculation (mAP@0.5, mAP@0.5:0.95)
│   ├── boxes.py          # Coordinate conversions (xywh <-> xyxy) & NMS
│   └── general.py        # Logging, seed fixing, checkpoint saving
│
├── data/
│   ├── __init__.py
│   ├── dataset.py        # PyTorch Dataset + custom collate_fn for variable batching
│   └── augmentations.py  # Mosaic, MixUp, random affine
│
├── tests/
│   ├── test_repconv.py   # Asserts out_multi == out_fused within 1e-5 tolerance
│   └── test_forward.py   # Validates feature map output shapes for dummy tensors
│
├── configs/
│   └── yolov7.yaml       # Hyperparameters, anchors, channel dimensions
│
├── train.py              # Training loop & intermediate evaluation
├── val.py                # Standalone evaluation on validation set
├── detect.py             # Single-image/video inference script with bounding box plots
└── README.md