# **AI-on-Edge-Devices: AI Model Deployment on Raspberry Pi**


## **Models 💻:**
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
