# **AI-on-Edge-Devices: AI Model Deployment on Raspberry Pi**

This repository demonstrates how to deploy trained AI models on edge devices, specifically the Raspberry Pi, with a comprehensive workflow covering training, optimization, and inference. The repository supports our research paper *['XYZ'](https://link.springer.com)* published in *[JOURNAL](https://google.com/)*, which aims to advance edge-based AI inference.

<p align="center">
    <img src="https://img.shields.io/badge/version-v0.0.0-rc0" alt="Version">
    <a href ="https://github.com/rashidrao-pk/AI_on_Edge_Devices/blob/main/LICENSE">
        <img src="https://img.shields.io/badge/License-MIT-green.svg" alt="License: MIT">
    </a>
    <a href="https://github.com/rashidrao-pk/">
        <img src="https://img.shields.io/github/contributors/rashidrao-pk/AI_on_Edge_Devices" alt="GitHub contributors">
    </a>
    <img src="https://img.shields.io/github/repo-size/rashidrao-pk/AI_on_Edge_Devices" alt="GitHub repo size">
    <a href="https://github.com/rashidrao-pk/">
        <img src="https://img.shields.io/github/commit-activity/t/rashidrao-pk/AI_on_Edge_Devices" alt="GitHub commit activity (branch)">
    </a>
    <a href="https://github.com/rashidrao-pk/AI_on_Edge_Devices/graphs/contributors">
        <img src="https://img.shields.io/github/contributors/rashidrao-pk/AI_on_Edge_Devices" alt="GitHub contributors">
    </a>
    <a href="https://github.com/rashidrao-pk/AI_on_Edge_Devices/issues?q=is%3Aissue+is%3Aclosed">
        <img src="https://img.shields.io/github/issues-closed/rashidrao-pk/AI_on_Edge_Devices" alt="GitHub closed issues">
    </a>
    <a href="https://github.com/rashidrao-pk/AI_on_Edge_Devices/issues">
        <img src="https://img.shields.io/github/issues/rashidrao-pk/AI_on_Edge_Devices" alt="GitHub issues">
    </a>
    <a href="https://github.com/rashidrao-pk/AI_on_Edge_Devices/pulls?q=is%3Apr+is%3Aclosed">
        <img src="https://img.shields.io/github/issues-pr-closed/rashidrao-pk/AI_on_Edge_Devices" alt="GitHub closed pull requests">
    </a>
    <a href="https://github.com/rashidrao-pk/AI_on_Edge_Devices/pulls">
        <img src="https://img.shields.io/github/issues-pr/rashidrao-pk/AI_on_Edge_Devices" alt="GitHub pull requests">
    </a>
    <img src="https://img.shields.io/github/last-commit/rashidrao-pk/AI_on_Edge_Devices" alt="GitHub last commit">
    <a href="https://github.com/rashidrao-pk/AI_on_Edge_Devices/watchers">
        <img src="https://img.shields.io/github/watchers/rashidrao-pk/AI_on_Edge_Devices?style=flat" alt="GitHub watchers">
    </a>
    <a href="https://github.com/rashidrao-pk/AI_on_Edge_Devices/forks">
        <img src="https://img.shields.io/github/forks/rashidrao-pk/AI_on_Edge_Devices?style=flat" alt="GitHub forks">
    </a>
    <a href="https://github.com/rashidrao-pk/AI_on_Edge_Devices/stargazers">
        <img src="https://img.shields.io/github/stars/rashidrao-pk/AI_on_Edge_Devices?style=flat" alt="GitHub Repo stars">
    </a>

</p>

<p align="center">
  <img src='docs/images/logo.png' width="30%" height="30%">
</p>

---

## **Project Overview:**
This repository consists of essential components for deploying AI on edge devices:
1. **Training a Deep Learning Model:** Implementations using various architectures, including VGG16, ResNet50, and custom models.
2. **Model Conversion:** Post-training model optimization and conversion to TensorFlow Lite for efficient deployment on resource-constrained devices.
3. **Inference and Real-Time Testing:** Perform inference on both local systems and Raspberry Pi, with cross-dataset testing.

---

## **Hardware and Setup Details:**

- **Model Training:**
  - **Primary Device:** Santech XN2 with NVIDIA RTX 4070 (8GB GPU), 16GB RAM, Intel Core i9 (13th Gen)
- **Model Inference on Edge Device:**
  - **Edge Device:** Raspberry Pi 4B (8GB RAM)
  - **Local Devices:** Integrated webcam (Chicony) and Realme5s phone camera (4GB RAM)

---

## **Datasets 📊:**
-  Publicaly Available Benchmarks
    - **`MNIST`**
    - **`CIFAR-10`**
    - **`CIFAR-100`**
-   [**Self Collected dataset**](datasets/inference).
    -   [`for CIFAR`](datasets/inference/cifar10/readme.md)

## **Deep Learning Models 💻:**
Models trained and optimized for edge deployment are available in this [repository](https://github.com/yasir). They include:
- `VGG16`, `ResNet50`, `Custom CNN` models

## **Model Training:**
To train a model from scratch:
 - Run the notebook **`N1_EDGE_AI_TRAIN.ipynb`** [here](codes/N1_EDGE_AI_TRAIN.ipynb).

---

## **Run Inference:**
To perform inference on trained models, use the following notebooks and scripts:

1. **Local Machine Inference:** Use integrated or mobile cameras to test model inference.
   - **Run Script for CIFAR-10:**
     ```bash
     python run_inference.py --dataset cifar10 --brightness 70
     ```
   - **Run Script for MNIST:**
     ```bash
     python run_inference.py --dataset mnist --brightness 70
     ```

2. **Edge Device Inference on Raspberry Pi:** Start with [this readme](docs/setup.md) to set up the Pi.

3. **Cross-Data Validation:** Inference is conducted on cross-dataset examples, including non-standard test images, such as sketches and cartoon CAD models.

---

## **Project Structure**
```
├── codes                     # Code for model training and inference
│   ├── N1_EDGE_AI_TRAIN.ipynb   # Notebook for model training
│   ├── N2_EDGE_AI_INFERENCE.ipynb # Notebook for model inference
├── docs                      # Documentation files and images
│   ├── N1_EDGE_AI_TRAIN
│   ├── N1_EDGE_AI_TRAIN
│   └── images
├── models                    # Saved models (trained and converted)
├── results                   # Saved results (trained and converted)   
└── README.md                 # Project documentation

```
---

---
## **Hardware 1 Configuration (Training)**
```cmd
Hardware Name   : Santech-XN2 Laptop
Hardware Memory : 16GB
GPU             : NVIDIA-RTX-4070
------------------------------------------------------------
Python version  : 3.9.18 (main, Sep 11 2023, 13:30:38) [MSC v.1916 64 bit (AMD64)]
TensorFlow version:       2.6.0
Keras version:            2.6.0
NumPy version:            1.23.0
System architecture:      ('64bit', 'WindowsPE')
Machine type:             AMD64
Processor:                Intel64 Family 6 Model 186 Stepping 2, GenuineIntel
Platform:                 Windows, 10.0.22631
Release:                  10
Version:                  10.0.22631
TensorFlow GPU Devices:
PhysicalDevice(name='/physical_device:GPU:0', device_type='GPU')
```

---
## **Edge Device Configuration (Inference)**

```cmd
Hardware Name   : Raspberry-Pi-4B
Hardware Memory : 8GB
GPU             : No GPU
------------------------------------------------------------
Python version:           3.9.2 (default, Feb 28 2021, 17:03:44) 
[GCC 10.2.1 20210110]
TensorFlow version:       2.13.0
Keras version:            2.13.1
NumPy version:            1.24.3
System architecture:      ('64bit', 'ELF')
Machine type:             aarch64
Processor:                
Platform:                 Linux, #1680 SMP PREEMPT Wed Sep 13 18:09:06 BST 2023
Release:                  6.1.53-v8+
Version:                  #1680 SMP PREEMPT Wed Sep 13 18:09:06 BST 2023
```
---

## **Keywords 🔍**
Edge Devices, Inference, Raspberry Pi, Deep Learning

---

## **Contributions✍️**

| Sr. No. | Author Name | Contributions | Affiliation | Google Scholar | 
| :--:    | :--:        | :--:        |:--:        | :--:           | 
| 1. | Muhammad Rashid | - |University of Turin, Computer Science Department, Italy | [Google Scholar](https://scholar.google.com/citations?user=F5u_Z5MAAAAJ&hl=en) | 
| 2. | Muhammad Yasir Shabir |- | University of Turin, Computer Science Department, Italy | [Google Scholar](https://scholar.google.com/citations?user=F5u_Z5MAAAAJ&hl=en) | 


---
## Roadmaps 
---

## **Contribution and Feedback**
Your feedback is valuable! Feel free to [open issues](https://github.com/rashidrao-pk/AI_on_Edge_Devices/issues) or [create pull requests](https://github.com/rashidrao-pk/AI_on_Edge_Devices/pulls). For direct communication, contact [Muhammad Rashid](mailto:muhammad.rashid@unito.it). 


---
Profile Views: <img src="https://api.visitorbadge.io/api/combined?path=https%3A%2F%2Fgithub.com%2Frashidrao-pk&label=Visitors&countColor=%23263759&style=flat" alt="Visitors">
