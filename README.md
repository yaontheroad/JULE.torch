# JULE: Joint Unsupervised Learning of Deep Representations and Image Clusters (PyTorch)

This is a PyTorch implementation of the CVPR 2016 paper [Joint Unsupervised Learning of Deep Representations and Image Clusters](https://arxiv.org/abs/1604.03628).

## Overview

This project performs joint unsupervised learning of deep CNN and image clusters. The intuition is that better image representation will facilitate clustering, while better clustering results will help representation learning. Given an unlabeled dataset, it will iteratively learn CNN parameters unsupervisedly and cluster images.

## Installation

1. Clone this repository:
```bash
git clone https://github.com/yourusername/JULE-pytorch.git
cd JULE-pytorch
```

2. Create a virtual environment and install dependencies:
```bash
python -m venv venv
source venv/bin/activate  # On Windows, use: venv\Scripts\activate
pip install -r requirements.txt
```

## Usage

1. Train the model:
```bash
python train.py --dataset USPS --eta 0.9
```

Key parameters:
- `--dataset`: Dataset name (USPS, MNIST-test, etc.)
- `--eta`: Unfolding rate (0.2 for face datasets, 0.9 for others)
- `--num_nets`: Number of parallel models to train
- `--use_fast`: Whether to use fast affinity updating algorithm
- `--batch_size`: Batch size for training
- `--learning_rate`: Base learning rate

## Project Structure

```
JULE-pytorch/
├── datasets/           # Dataset loading and processing
├── models/            # Network architecture definitions
├── criterions/        # Loss functions
├── clustering/        # Clustering algorithms
├── affinity/         # Affinity computation
├── evaluate/         # Evaluation metrics
└── utils/            # Utility functions
```

## Citation

If you find this code useful in your research, please consider citing:

```bibtex
@inproceedings{yangCVPR2016joint,
    Author = {Yang, Jianwei and Parikh, Devi and Batra, Dhruv},
    Title = {Joint Unsupervised Learning of Deep Representations and Image Clusters},
    Booktitle = {IEEE Conference on Computer Vision and Pattern Recognition (CVPR)},
    Year = {2016}
}
```

## License

This code is released under the MIT License.
