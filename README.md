# PR-CapsNet: Pseudo-Riemannian Capsule Network with Adaptive Curvature Routing for Graph Learning

[![arXiv](https://img.shields.io/badge/arXiv-2512.08218-b31b1b.svg)](https://arxiv.org/abs/2512.08218)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python](https://img.shields.io/badge/Python-3.11%2B-blue)](https://www.python.org/)
[![PyG](https://img.shields.io/badge/PyG-2.0%2B-green)](https://www.pyg.org/)

**[English](#english-description) | [简体中文](#chinese-description)**

<a name="english-description"></a>

## 📖 Introduction

This repository contains the implementation of the paper: **"PR-CapsNet: Pseudo-Riemannian Capsule Networks"**.

**Abstract:**  
Current capsule networks predominantly rely on Euclidean or hyperbolic spaces with **fixed curvature**, struggling to effectively model complex geometric structures characterized by **coexisting hierarchies, clusters, and cycles** in real-world graph data. To address this, we propose **PR-CapsNet**, which generalizes capsule routing mechanisms to **Pseudo-Riemannian manifolds with learnable curvature** for the first time. Specifically, we construct a **time-space decoupled tangent space routing** via diffeomorphic transformations and introduce an **Adaptive Curvature Routing (ACR)** mechanism based on local manifold properties. Our model achieves **State-of-the-Art (SOTA) performance** on multiple node and graph classification benchmarks while **significantly reducing computational overhead**.

📄 **Paper:** [arXiv:2512.08218](https://arxiv.org/abs/2512.08218)

---

## 📂 Project Structure

This is a self-contained implementation. All model architectures, geometric operations, and training logic are included in a single file for ease of reproducibility.

```text
PR-CapsNet/
├── PR-CapsNet.py          # The complete implementation
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
You can download dataset as follow:
*   **Automatic Download**: When you run the code for the first time, it will attempt to automatically download datasets (Cora, Citeseer, CoauthorCS, etc.) via PyTorch Geometric into a `./data` folder.
*   **Manual Download**: If you encounter network issues, please manually download the "Planetoid" or "Coauthor" datasets and place the raw files into the `./data` directory following PyG conventions.

---

## 🏃 Usage

To run the experiments:

```bash
python PR-CapsNet.py
```

You can modify the `ModelConfig` class inside `PR-CapsNet.py` to change hyperparameters (e.g., dimensions, dropout, dataset):

```python
class ModelConfig:
    s_dim: int = 9                  # Space-like dimensions
    t_dim: int = 9                  # Time-like dimensions
    seed: int = 2903                 # Random seed 2903
```

---

## 🖊️ Citation

If you find our work useful in your research, please consider citing:

```bibtex
@misc{qin2025prcapsnetpseudoriemanniancapsulenetwork,
      title={PR-CapsNet: Pseudo-Riemannian Capsule Network with Adaptive Curvature Routing for Graph Learning}, 
      author={Ye Qin and Jingchao Wang and Yang Shi and Haiying Huang and Junxu Li and Weijian Liu and Tinghui Chen and Jinghui Qin},
      year={2025},
      eprint={2512.08218},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2512.08218}, 
}
```

---
<br>

<a name="chinese-description"></a>

## 📖 项目简介

本仓库包含论文 **"PR-CapsNet: Pseudo-Riemannian Capsule Networks"** (PR-CapsNet: 伪黎曼胶囊网络) 的实现。

**摘要：**
现有胶囊网络多基于**固定曲率**的欧氏或双曲空间，难以有效建模真实图数据中同时存在的**层次、聚类与环状**等复杂几何结构。为此，我们提出 **PR-CapsNet**，首次将胶囊路由机制拓展至**可学习曲率的伪黎曼流形**：通过微分同胚变换构建**时空-空间解耦的切空间路由**，并引入基于局部流形性质的**自适应曲率融合机制**，在多个节点与图分类基准上取得 **SOTA**性能，同时显著降低计算开销。

📄 **论文链接:** [arXiv:2512.08218](https://arxiv.org/abs/2512.08218)

---

## 📂 项目结构

本项目采用单文件实现，便于阅读与复现。所有核心逻辑均包含在 `PR-CapsNet.py` 中。

```text
PR-CapsNet/
├── PR-CapsNet.py          # 完整代码实现
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
数据集可以通过以下操作下载。
*   **自动下载**：首次运行代码时，程序会通过 PyTorch Geometric 接口自动下载所需数据集（Cora, Citeseer 等）并保存在 `./data` 目录下。
*   **手动下载**：如果您的网络环境受限，请自行下载 Planetoid 或 Coauthor 数据集，并按 PyG 格式要求放入 `./data` 目录。

---

## 🏃 运行指南

直接运行脚本即可开始训练并得到实验结果：

```bash
python PR-CapsNet.py
```

如需修改超参数（如维度、数据集、Dropout率），请直接在 `PR-CapsNet.py` 中的 `ModelConfig` 部分进行修改。
```python
class ModelConfig:
    s_dim: int = 9                  # 类空间维度
    t_dim: int = 9                  # 类时间维度
    seed: int = 2903                 # 随机种子2903
```
---

## 📜 引用

如果您觉得本工作对您的研究有帮助，请引用我们的论文：

```bibtex
@misc{qin2025prcapsnetpseudoriemanniancapsulenetwork,
      title={PR-CapsNet: Pseudo-Riemannian Capsule Network with Adaptive Curvature Routing for Graph Learning}, 
      author={Ye Qin and Jingchao Wang and Yang Shi and Haiying Huang and Junxu Li and Weijian Liu and Tinghui Chen and Jinghui Qin},
      year={2025},
      eprint={2512.08218},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2512.08218}, 
}
```
