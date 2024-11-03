# **AI-on-Edge-Devices: AI Model Deployment on Raspberry Pi**


## **Models 💻:**
Models trained and optimized for edge deployment are available in this [repository](https://github.com/yasir). They include:
- `VGG16`, `ResNet50`, `Custom CNN` models

## **Model Training:**
To train a model from scratch:
 - Run the notebook **`N1_EDGE_AI_TRAIN.ipynb`** [here](codes/N1_EDGE_AI_TRAIN.ipynb).
---

## Test Model
Trained model are testing in following settings:
1. Base model without any optimization
2. Quantized model on Same Same Machine  (used for training)
3. Quantized model on Edge-Device (Resberry-Pi 4B)


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

---
## **Project Structure**
```
├── codes                               # Code for model training and inference
│   ├── N1_EDGE_AI_TRAIN.ipynb          # Notebook for model training
│   ├── N2_EDGE_AI_INFERENCE.ipynb      # Notebook for model inference
├── models                              # Saved models (trained and converted)
├── results                             # Saved results (trained and converted)   
└── README.md                           # Codes documentation

```
---



---
## **Hardware 1 Details used for Tarining**
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
## **Edge Device Setup**

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
No GPU detected by TensorFlow
```
---