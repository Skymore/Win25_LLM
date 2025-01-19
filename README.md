# Win25_LLM Project

This repository contains deep learning and machine learning experiments and implementations.

## Environment Setup

### Using Mamba (Recommended)
```bash
# Create environment
mamba env create -f environment.yml

# Activate environment
mamba activate llm596
```

### Manual Installation
If you prefer to install packages manually:
```bash
mamba create -n llm596 python=3.11
mamba activate llm596

# Install PyTorch with CUDA support
mamba install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia

# Install TensorFlow
mamba install tensorflow-gpu

# Install other dependencies
mamba install -c conda-forge numpy matplotlib jupyter pillow
pip install openai
```

## Project Structure
```
.
├── README.md
├── environment.yml
└── notebooks/
    └── Coding Assignment 1-1.ipynb
```

## GPU Support
- The environment is configured to support GPU acceleration for both PyTorch and TensorFlow
- Verify GPU support:
  ```python
  import torch
  print("PyTorch GPU:", torch.cuda.is_available())
  
  import tensorflow as tf
  print("TensorFlow GPU:", len(tf.config.list_physical_devices('GPU')) > 0)
  ```

## License
This project is licensed under the MIT License - see the LICENSE file for details. 