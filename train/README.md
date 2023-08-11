## Training part of our LPCV 2023 solution

### Training codebase

[United Perception](https://github.com/ModelTC/United-Perception) with [LPCV Plugin]()

### Environment

#### Hardware
- 2x Xeon Gold CPU
- 8x V100 32GB

#### Base Software
- CentOS
- GPU driver 525.85.12
- Slurm Scheduler

#### Main Python Packages
- Python==3.6.9
- torch==1.8.1+cuda112.cudnn8.1.0
- torchvision==0.9.0a0+8fb5838
- einops==0.4.1
- numba==0.53.0 
- numpy==1.17.5
- onnx==1.6.0
- opencv-python==4.1.1.26

### Training Step
```bash
git clone https://github.com/ModelTC/United-Perception
git clone xxx

# Assume your Python environment is ready
# install UP as doc: https://modeltc-up.readthedocs.io/en/latest/index.html
./easy_setup.sh

# Training costs about 3 hours 
sh scripts/train.sh 8 lpcv_train.yaml

# Convert to onnx
sh scripts/convert_to_onnx.sh 1 lpcv_train.yaml

# Go to ../README.md for next steps 
```

### Checkpoints 
- `models/pretrain_weight.pth` includes pretrain weights, which only contain backbone weights since it was trained on classification datasets.

- `models/ckpt_weight_with_ema.pth` is our trained weights, which is the same as our submission.

### Training log
`tf_logs/events.out.tfevents.1690350062.log` 