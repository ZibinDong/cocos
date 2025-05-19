# Conditioning Matters: Training Diffusion Policies is Faster Than You Think

[Paper](https://arxiv.org/abs/2505.11123) | [Checkpoints]()

Official implementation of the paper "Conditioning Matters: Training Diffusion Policies is Faster Than You Think". We provide a PyTorch implementation of the proposed method for training diffusion policies on the LIBERO benchmark.

## Requirements

**1. CleanDiffuser**

This repository is based on [CleanDiffuser](https://github.com/CleanDiffuserTeam/CleanDiffuser/tree/lightning). Please install its `lightning` branch:

```bash
conda create -n cocos python==3.10
conda activate cocos

# Any torch>1.0.0 that is compatible with your CUDA version. For example:
conda install pytorch==2.2.2 torchvision==0.17.2 pytorch-cuda=12.1 -c pytorch -c nvidia

git clone https://github.com/CleanDiffuserTeam/CleanDiffuser.git
cd CleanDiffuser
git checkout lightning
pip install -e .
```

Than follow the instructions [here](https://github.com/CleanDiffuserTeam/CleanDiffuser/blob/lightning/cleandiffuser/env/README.md) to install the LIBERO dependencies and prepare the dataset.

Then overwrite the `DATASET_PATH` in `autoencoder.py` and `cocos.py` with the path you save the zarr file to.

## Usage

Overwrite the `SAVE_PATH` in `autoencoder.py` and `cocos.py` with the path you want to save the checkpoints to.

**1. Use Pretrained Checkpoints**

Download the pretrained checkpoints from [here](https://1drv.ms/u/c/ba682474b24f6989/EdyFZAAU47BPtZohTYDMHfwBeWB51Z8wSupl6z-iqfqBuw?e=kjBpzM) and unzip the file to `SAVE_PATH`. Then run the following commands to evaluate the pretrained diffusion policy:

```bash
python cocos.py --mode inference --task_suite libero_goal --task_id 0
```

**2. Train from Scratch**

Train an autoencoder as the condition-dependent source distribution:
```bash
python autoencoder.py --task_suite libero_goal
```
Train a diffusion policy:
```bash
python cocos.py --mode training --task_suite libero_goal
```
Then evaluate the diffusion policy:
```bash
python cocos.py --mode inference --task_suite libero_goal --task_id 0
```

> *Note:* As discussed in the paper, the condition-dependent source distribution $q(z|c)$ can take any possible form. For simplicity, we use a fixed-std Gaussian and train it using an autoencoding objective. We encourage users to explore other forms of $q(z|c)$ by changing code in `autoencoder.py`.

## Citation

If you find this code useful, please consider citing our paper:

```bibtex
@inproceedings{dong2025cocos,
    title={Conditioning Matters: Training Diffusion Policies is Faster Than You Think}, 
    author={Zibin Dong, Yicheng Liu, Yinchuan Li, Hang Zhao, Jianye Hao},
    year={2025},
    booktitle={arXiv, preprint arXiv:2505.11123},
}
```