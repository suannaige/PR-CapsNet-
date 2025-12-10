# PR-CapsNet: Pseudo-Riemannian Capsule Networks

[![arXiv](https://img.shields.io/badge/arXiv-2512.08218-b31b1b.svg)](https://arxiv.org/abs/2512.08218)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python](https://img.shields.io/badge/Python-3.8%2B-blue)](https://www.python.org/)
[![PyG](https://img.shields.io/badge/PyG-2.0%2B-green)](https://www.pyg.org/)

**[English](#english-description) | [简体中文](#chinese-description)**

<a name="english-description"></a>

## 📖 Introduction

This repository contains the official PyTorch implementation of the paper: **"PR-CapsNet: Pseudo-Riemannian Capsule Networks"**.

**Abstract:**  
PR-CapsNet introduces a novel framework that generalizes capsule networks to **Pseudo-Riemannian manifolds**. By leveraging indefinite signatures (time-like and space-like dimensions), our model effectively captures complex hierarchies and heterogeneous relationships in graph data. We introduce **Adaptive Curvature Routing (ACR)** and a numerically stable implementation of the Exponential and Logarithmic maps to ensure robust training in non-Euclidean spaces.

📄 **Paper:** [arXiv:2512.08218](https://arxiv.org/abs/2512.08218)

---

## 📂 Project Structure

This is a self-contained implementation. All model architectures, geometric operations, and training logic are included in a single file for ease of reproducibility.

```text
PR-CapsNet/
├── main.py          # The complete implementation (Model, Training, Config)
└── README.md        # Documentation
```

*Note: The `data/` directory will be generated automatically upon running the script.*

---

## 🛠️ Installation & Data

### 1. Dependencies
We recommend using Conda. Ensure you have `torch` and `torch_geometric` installed.

```bash
# Basic setup
pip install numpy pandas

# Install PyTorch (Adjust CUDA version as needed)
pip install torch --index-url https://download.pytorch.org/whl/cu118

# Install PyTorch Geometric and torch-scatter (Essential!)
pip install torch_geometric
pip install torch_scatter -f https://data.pyg.org/whl/torch-2.0.0+cu118.html
```

### 2. Data Preparation
The repository does **not** include datasets to keep the file size minimal.
*   **Automatic Download**: When you run the code for the first time, it will attempt to automatically download datasets (Cora, Citeseer, CoauthorCS, etc.) via PyTorch Geometric into a `./data` folder.
*   **Manual Download**: If you encounter network issues, please manually download the "Planetoid" or "Coauthor" datasets and place the raw files into the `./data` directory following PyG conventions.

---

## 🏃 Usage

To run the experiments (reproducing the results in the paper):

```bash
python main.py
```

You can modify the `ModelConfig` class inside `main.py` to change hyperparameters (e.g., dimensions, dropout, dataset):

```python
# Configuration in main.py
config = ModelConfig(
    dataset_name='Cora',
    s_dim=9, t_dim=9,
    learnable_curvature=True
)
```

---

## 🖊️ Citation

If you find our work useful in your research, please consider citing:

```bibtex
@article{prcapsnet2025,
  title={PR-CapsNet: Pseudo-Riemannian Capsule Networks},
  author={Your Name and Co-authors},
  journal={arXiv preprint arXiv:2512.08218},
  year={2025}
}
```

---
<br>

<a name="chinese-description"></a>

## 📖 项目简介 (Chinese)

本仓库包含论文 **"PR-CapsNet: Pseudo-Riemannian Capsule Networks"** (PR-CapsNet: 伪黎曼胶囊网络) 的官方 PyTorch 实现。

**摘要：**
PR-CapsNet 提出了一种全新的框架，将胶囊网络推广至 **伪黎曼流形（Pseudo-Riemannian Manifolds）**。通过利用非定签名的度量空间（包含时间维和空间维），我们的模型能够有效地捕捉图数据中复杂的层级结构和异质关系。我们提出了 **自适应曲率路由（ACR）** 机制，并实现了一套数值稳健的指数映射与对数映射算法，确保模型在非欧空间训练时的稳定性。

📄 **论文链接:** [arXiv:2512.08218](https://arxiv.org/abs/2512.08218)

---

## 📂 项目结构

本项目采用单文件实现，便于阅读与复现。所有核心逻辑（几何计算、模型结构、训练引擎）均包含在 `main.py` 中。

```text
PR-CapsNet/
├── main.py          # 完整代码实现
└── README.md        # 项目说明
```

*注：`data/` 目录将在代码运行时自动生成。*

---

## 🛠️ 安装与数据

### 1. 环境依赖
请确保安装了 PyTorch 和 PyTorch Geometric。

```bash
# 基础依赖
pip install numpy pandas

# 安装 PyTorch (请根据您的 CUDA 版本调整)
pip install torch --index-url https://download.pytorch.org/whl/cu118

# 关键依赖：torch_scatter (必须安装)
pip install torch_geometric
pip install torch_scatter -f https://data.pyg.org/whl/torch-2.0.0+cu118.html
```

### 2. 数据集准备
为了保持仓库轻量化，我们**不提供**原始数据文件。
*   **自动下载**：首次运行代码时，程序会通过 PyTorch Geometric 接口自动下载所需数据集（Cora, Citeseer 等）并保存在 `./data` 目录下。
*   **手动下载**：如果您的网络环境受限，请自行下载 Planetoid 或 Coauthor 数据集，并按 PyG 格式要求放入 `./data` 目录。

---

## 🏃 运行指南

直接运行脚本即可开始训练并复现论文中的实验结果：

```bash
python main.py
```

如需修改超参数（如维度、数据集、Dropout率），请直接在 `main.py` 中的 `ModelConfig` 部分进行修改。

---

## 📜 引用

如果您觉得本工作对您的研究有帮助，请引用我们的论文：

```bibtex
@article{prcapsnet2025,
  title={PR-CapsNet: Pseudo-Riemannian Capsule Networks},
  author={Your Name and Co-authors},
  journal={arXiv preprint arXiv:2512.08218},
  year={2025}
}
```
