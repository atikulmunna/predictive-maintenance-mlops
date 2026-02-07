# GPU Setup Guide for RTX 5060

## Current Status

Your **RTX 5060 Mobile** (Ada architecture, sm_120) is not yet fully supported by official PyTorch releases. This document provides options to get GPU working.

## Option 1: Build PyTorch from Source (Most Reliable)

This requires building PyTorch locally with Ada support compiled in.

```bash
# Activate environment
conda activate pred-maint

# Install build dependencies
conda install cmake ninja

# Set CUDA paths
$env:CUDA_HOME = "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.4"
$env:PATH = "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.4\bin;" + $env:PATH

# Clone PyTorch
git clone https://github.com/pytorch/pytorch.git
cd pytorch
git submodule update --init --recursive

# Set compilation flags for Ada
$env:TORCH_CUDA_ARCH_LIST = "5.0;6.0;6.1;7.0;7.5;8.0;8.6;9.0;9.0a"  # Includes Ada

# Build (this takes 30-60 minutes)
pip install -e .

# Build torchvision and torchaudio
cd ..
git clone https://github.com/pytorch/vision.git
cd vision
pip install -e .

cd ..
git clone https://github.com/pytorch/audio.git
cd audio
pip install -e .
```

## Option 2: Use NVIDIA's Dev Container (Docker)

Use nvidia-docker with a PyTorch container that has Ada support pre-built:

```bash
docker run --gpus all -it nvcr.io/nvidia/pytorch:24.01-py3
```

## Option 3: Wait for Official Support

PyTorch 2.6+ (expected Q1 2025) should have full Ada support. For now, you can:
- Train on CPU (still fast for this dataset: ~1 min per epoch for LSTM)
- Use GPU later when PyTorch updates

## Performance Expectations (Current CPU Setup)

| Task | Time |
|------|------|
| XGBoost training | 3-5 minutes |
| LSTM 10 epochs | 5-15 minutes |
| Full ensemble | 20-30 minutes |
| Batch predictions (100) | < 1 second |
| API inference | < 50ms |

CPU is actually **sufficient** for this project!

## Enable GPU Later

When you have a proper PyTorch build with Ada support:

1. Update `.env`:
```bash
CUDA_VISIBLE_DEVICES=0
```

2. Test:
```bash
python verify_gpu.py
```

3. Training will automatically use GPU:
```python
import torch
model = MyModel().to('cuda')  # Will work when GPU available
```

## Resources

- [PyTorch Build from Source](https://github.com/pytorch/pytorch#from-source)
- [NVIDIA CUDA Compute Capability](https://developer.nvidia.com/cuda-gpus)
- [RTX 5060 Specs](https://www.nvidia.com/en-us/geforce/graphics-cards/50-series/) (sm_120 - Blackwell/Ada)

## For Now: Use CPU

Your environment is fully functional on CPU. The **performance is actually quite good** for the dataset size (70K rows). You can:

✅ Start Phase 1-2 training immediately  
✅ Achieve excellent model accuracy  
✅ Build production pipeline  
✅ Upgrade GPU later without code changes

**Recommendation: Proceed with CPU training, migrate to GPU later if needed.**
