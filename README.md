
# SPFD
This is the official implementation of the paper "Beyond Duality: A Hybrid Framework of Leveraging Shared and Private
Features for RGB-Event Object Detection".

<div align="center">
  <img src="figures/framework.png"/>
</div><br/>

## Installation

We tested our code with `Python=3.8.39, PyTorch=1.12.0, CUDA=11.3`. Please install PyTorch first according to [official instructions](https://pytorch.org/get-started/previous-versions/). Our code is based on [detrex](https://github.com/IDEA-Research/detrex/tree/main). Please refer to the [installation](https://detrex.readthedocs.io/en/latest/tutorials/Installation.html) of detrex.

Example conda environment setup：

```bash
# Create a new virtual environment
conda create -n SPFD python=3.8 -y
conda activate SPFD

# Install PyTorch
pip install torch==1.12.0+cu113 torchvision==0.13.0+cu113 torchaudio==0.12.0 --extra-index-url https://download.pytorch.org/whl/cu113

# initialize the detectron2 submodule
git init
git submodule init
git submodule update

# Install detectron2
python -m pip install 'git+https://github.com/facebookresearch/detectron2.git'

# Under your working directory
git clone https://github.com/git-KeYw/SPFD.git
cd SPFD
pip install -r requirements.txt
```
## Configure dataset path
```bash
#Configure dataset root path for evaluation in
/detrex/data/datasets/register_pku.py #Line 454

# build an editable version of detrex
pip install -e .
mv ./projects/midetr/configs/data/pku.py ./detrex/config/configs/common/data
```

## Run

### train

You can train our models with the following commands.

commands for pku_davis_sod dataset:
```sh
python tools/train_net.py   --num-gpus 2   --config-file projects/midetr/configs/midetr-resnet/SPFD_pku.py
```

### Evaluation

You can evaluate our pretrained models with the following commands.

commands for pku_davis_sod dataset:
```sh
python tools/test_net.py   --num-gpus 2   --config-file projects/midetr/configs/midetr-resnet/SPFD_pku.py
```


