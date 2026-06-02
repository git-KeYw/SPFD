# SPFD

<div align="center">

# Beyond Duality: A Hybrid Framework of Leveraging Shared and Private Features for RGB-Event Object Detection

**CVPR 2026**

[![Paper](https://img.shields.io/badge/Paper-CVF-blue)](https://openaccess.thecvf.com/content/CVPR2026/html/Wang_Beyond_Duality_A_Hybrid_Framework_of_Leveraging_Shared_and_Private_CVPR_2026_paper.html)
[![Conference](https://img.shields.io/badge/CVPR-2026-red)](https://cvpr.thecvf.com/)
[![Code](https://img.shields.io/badge/Code-SPFD-green)](https://github.com/git-KeYw/SPFD)

</div>

---

## 📄 Paper

Our paper is available on the CVF Open Access:

[**Beyond Duality: A Hybrid Framework of Leveraging Shared and Private Features for RGB-Event Object Detection**](https://openaccess.thecvf.com/content/CVPR2026/html/Wang_Beyond_Duality_A_Hybrid_Framework_of_Leveraging_Shared_and_Private_CVPR_2026_paper.html)

This repository provides the official implementation of this paper.

---

## 🎉 News

- **[2026.06]** 🎉 Our paper has been accepted by **CVPR 2026**.
- **[2026.06]** 🚀 The official implementation of **SPFD** is released.

---

## 🧩 Framework

SPFD is designed for RGB-Event object detection. It leverages both shared and private modality-specific features to better exploit the complementary information between RGB frames and event streams.

<div align="center">
  <img src="figures/framework.png" width="95%"/>
</div>

---

## 🛠️ Installation

We tested our code with `Python 3.8`, `PyTorch 1.12.0`, and `CUDA 11.3`.

Our code is based on [detrex](https://github.com/IDEA-Research/detrex/tree/main). Please also refer to the official [detrex installation guide](https://detrex.readthedocs.io/en/latest/tutorials/Installation.html).

```bash
# Create a new virtual environment
conda create -n SPFD python=3.8 -y
conda activate SPFD

# Install PyTorch
pip install torch==1.12.0+cu113 torchvision==0.13.0+cu113 torchaudio==0.12.0 --extra-index-url https://download.pytorch.org/whl/cu113

# Clone this repository
git clone https://github.com/git-KeYw/SPFD.git
cd SPFD

# Initialize the detectron2 submodule
git init
git submodule init
git submodule update

# Install detectron2
python -m pip install 'git+https://github.com/facebookresearch/detectron2.git'

# Install other dependencies
pip install -r requirements.txt

# Build an editable version of detrex
pip install -e .
```

---

## 📁 Dataset Configuration

Please configure the dataset root path for PKU-DAVIS-SOD in:

```text
detrex/data/datasets/register_pku.py
```

Specifically, modify the dataset path around:

```text
Line 454
```

Then move the dataset config file:

```bash
mv ./projects/midetr/configs/data/pku.py ./detrex/config/configs/common/data
```

---

## 🚆 Training

You can train SPFD on the PKU-DAVIS-SOD dataset with:

```bash
python tools/train_net.py \
  --num-gpus 2 \
  --config-file projects/midetr/configs/midetr-resnet/SPFD_pku.py
```

---

## 🔍 Evaluation

You can evaluate the pretrained model with:

```bash
python tools/test_net.py \
  --num-gpus 2 \
  --config-file projects/midetr/configs/midetr-resnet/SPFD_pku.py
```

---

## 📌 Citation

If you find this work useful for your research, please consider citing our paper:

```bibtex
@inproceedings{wang2026beyond,
  title={Beyond Duality: A Hybrid Framework of Leveraging Shared and Private Features for RGB-Event Object Detection},
  author={Wang, Keyao and Liu, Shuai and Shi, Hengda and Shi, Lukui and Chen, Haiyong},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  year={2026}
}
```

---

## 🙏 Acknowledgements

This project is built upon the excellent open-source projects:

- [detrex](https://github.com/IDEA-Research/detrex)
- [Detectron2](https://github.com/facebookresearch/detectron2)
  
We sincerely thank the authors for their great work.

---

## 📜 License

This project is released for academic research purposes only.
