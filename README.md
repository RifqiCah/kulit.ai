# 🩺 Kulit.ai: Cloud-Based Skin Condition Classifier

![AWS](https://img.shields.io/badge/AWS-SageMaker-orange?logo=amazon-aws)
![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?logo=pytorch)
![Streamlit](https://img.shields.io/badge/Frontend-Streamlit-FF4B4B?logo=streamlit)
![Model](https://img.shields.io/badge/Model-EfficientNetV2-success)

**Kulit.ai** is a cloud-native deep learning application designed to classify skin conditions from images.

Unlike simple local deployments, this project demonstrates a **production-ready MLOps pipeline**. The model training and inference are decoupled: the heavy lifting is handled by **AWS SageMaker endpoints**, while the user interface is a lightweight **Streamlit** app interacting via AWS SDK (Boto3).

## 🚀 Key Features

* **☁️ AWS SageMaker Deployment:** The model is not running locally on the web server but is served via a persistent, scalable SageMaker Endpoint using `PyTorchPredictor`.
* **🧠 EfficientNetV2 Architecture:** Utilizes Transfer Learning on the EfficientNetV2 backbone, fine-tuned on a custom dataset of 3,152 images.
* **🔄 Decoupled Architecture:** Separates the frontend logic from the inference engine, mimicking real-world microservices patterns.
* **🛡️ Robust Preprocessing:** Implements identical tensor transformation logic in both the training pipeline and the Streamlit inference engine to prevent training-serving skew.

## 🏗️ Architecture

The system follows a cloud-client architecture:

```mermaid
graph LR
    User[User Upload Image] -->|Frontend| A[Streamlit App]
    
    subgraph Local_Processing
    A -->|Resize & Normalize| B(Tensor Preprocessing)
    end
    
    subgraph AWS_Cloud
    B -->|Boto3 Request| C[AWS API Gateway / Invocation]
    C -->|Inference| D[AWS SageMaker Endpoint]
    D -- Load Model --> E[EfficientNetV2 Artifacts]
    D -->|Prediction JSON| C
    end
    
    C -->|Return Result| A
    A -->|Display| User
