# 🚀 AR-TTA: A Simple Method for Real-World Continual Test-Time Adaptation

[![Paper](https://img.shields.io/badge/Paper-arXiv:2309.10109-red)](https://arxiv.org/pdf/2309.10109)
[![Conference](https://img.shields.io/badge/Conference-BMVC%202024-blue)](https://bmva-archive.org.uk/bmvc/2024/papers/Paper_25/paper.pdf)
[![Conference](https://img.shields.io/badge/Conference-ICCVW%202023-blue)](https://openaccess.thecvf.com/content/ICCV2023W/VCL/html/Sojka_AR-TTA_A_Simple_Method_for_Real-World_Continual_Test-Time_Adaptation_ICCVW_2023_paper.html)

> **Authors:** Damian Sójka, Sebastian Cygert, Bartłomiej Twardowski, Tomasz Trzciński

A simple yet effective method for real-world continual test-time adaptation that enables models to continuously adapt to changing environments during inference.

---

## 📋 Table of Contents

- [Environment Setup](#-environment-setup)
- [Dataset Preparation](#-dataset-preparation)
- [Running Experiments](#-running-experiments)
- [Citation](#-citation)

---

## 🛠 Environment Setup

1. **Clone the repository**
   ```bash
   git clone https://github.com/dmn-sjk/AR-TTA.git
   cd AR-TTA
   ```

2. **Create conda environment**
   ```bash
   conda env create -f environment.yml
   ```

3. **Activate the environment**
   ```bash
   conda activate artta
   ```

---

## 📊 Dataset Preparation

### CIFAR-10 and CIFAR-10-C Datasets

1. Download the CIFAR-10 dataset from [this link](https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz) and decompress the files.

2. Download CIFAR-10-C from [this link](https://zenodo.org/records/2535967/files/CIFAR-10-C.tar?download=1) and decompress the files.

3. **Preprocess the datasets:**
   ```bash
   python utils/process_cifar.py cifar-10c <path_to_CIFAR10C_dataset> <path_to_CIFAR10_dataset>
   ```

### ImageNet and ImageNet-C Datasets
1. Download [ImageNet](https://www.image-net.org/download.php) and [ImageNet-C](https://zenodo.org/record/2235448#.Yj2RO_co_mF).

### SHIFT Dataset

1. Download the dataset using the following command:

   ```bash
   python datasets/shift_dev/download.py --view front --group "[img, det_2d]" --split all --framerate images <TARGET_DIR>
   ```

### CLAD Dataset

1. Download and decompress the labeled SODA10M dataset following the [official instructions](https://soda-2d.github.io/download.html#instructions)

---

## 🤖 Source Model Checkpoints

CLAD and SHIFT dataset checkpoints for each seed are available [here](https://drive.google.com/drive/folders/1alLSzB1kJCfubmi9AB8iYmW7DvGj1pEa?usp=sharing).

Checkpoints for CIFAR-10 and ImageNet datasets are automatically downloaded by the [robustbench](https://github.com/RobustBench/robustbench) library.

---

## 🧪 Running Experiments

### Running CTTA

Run experiments using the `run.py` script. Adjust the configuration inside the script to modify arguments:

```bash
python run.py
```

### Running Source Training with example arguments

```bash
python train_source.py \
    --data_root /raid/NFS_SHARE/datasets \
    --dataset shift \
    --cuda 0 \
    --num_workers 5 \
    --run_name source_size32 \
    --scheduler_gamma 0.85 \
    --model wideresnet28
```
---

## 📄 Citation

If you use this work in your research, please cite our paper:

```bibtex
@inproceedings{Sójka_2024_BMVC,
    author    = {Damian Sójka and Bartłomiej Twardowski and Tomasz Trzcinski and Sebastian Cygert},
    title     = {AR-TTA: A Simple Method for Real-World Continual Test-Time Adaptation},
    booktitle = {35th British Machine Vision Conference 2024, {BMVC} 2024, Glasgow, UK, November 25-28, 2024},
    publisher = {BMVA},
    year      = {2024},
    url       = {https://bmva-archive.org.uk/bmvc/2024/papers/Paper_25/paper.pdf}
}
```