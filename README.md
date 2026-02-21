# Cats vs Dogs Classification — End-to-End MLOps Pipeline

## Project Overview

This project implements a **complete end-to-end MLOps pipeline** for a **binary image classification system (Cats vs Dogs)** designed for a **pet adoption platform**.

The system:

* trains an image classification model
* tracks experiments and artifacts
* exposes predictions via REST API
* containerizes the service
* automates CI/CD deployment
* supports monitoring and logging

The implementation uses **open-source tools** and follows **production-grade ML engineering practices**.

---

# System Architecture Diagram

```
                    ┌────────────────────────┐
                    │   Kaggle Dataset       │
                    │  Cats vs Dogs Images   │
                    └────────────┬───────────┘
                                 │
                                 ▼
                    ┌────────────────────────┐
                    │   Data Preprocessing   │
                    │ Resize 224x224 RGB    │
                    │ Augmentation + Split   │
                    └────────────┬───────────┘
                                 │
                                 ▼
                    ┌────────────────────────┐
                    │   Model Training       │
                    │  CNN / Transfer Model  │
                    └────────────┬───────────┘
                                 │
                                 ▼
                    ┌────────────────────────┐
                    │   MLflow Tracking      │
                    │ metrics, params, model │
                    └────────────┬───────────┘
                                 │
                                 ▼
                    ┌────────────────────────┐
                    │ Saved Model Artifact   │
                    │        model.h5        │
                    └────────────┬───────────┘
                                 │
                                 ▼
                    ┌────────────────────────┐
                    │ FastAPI Inference API  │
                    │ /health  /predict      │
                    └────────────┬───────────┘
                                 │
                                 ▼
                    ┌────────────────────────┐
                    │     Docker Image       │
                    │  Containerized Service │
                    └────────────┬───────────┘
                                 │
                                 ▼
                    ┌────────────────────────┐
                    │   CI Pipeline          │
                    │ GitHub Actions         │
                    │ test + build + push    │
                    └────────────┬───────────┘
                                 │
                                 ▼
                    ┌────────────────────────┐
                    │   Deployment Target    │
                    │ Docker Compose / K8s   │
                    └────────────┬───────────┘
                                 │
                                 ▼
                    ┌────────────────────────┐
                    │ Monitoring & Logs      │
                    │ latency, requests,     │
                    │ predictions            │
                    └────────────────────────┘
```

---

# Technology Stack

### Machine Learning

* TensorFlow / Keras
* NumPy, Pillow
* MLflow (experiment tracking)

> **Note:** the `mlflow-skinny` package installs as a lightweight subset and does **not** include
> the full tracking store API. Make sure you install the complete `mlflow` distribution (e.g.
> `pip install mlflow==2.15.1`) so that modules such as `mlflow.store` are available.

### Backend / Serving

* FastAPI REST service
* Uvicorn ASGI server

### MLOps & DevOps

* Git (code versioning)
* DVC / Git-LFS (dataset versioning optional)
* Docker (containerization)
* Docker Compose / Kubernetes (deployment)
* Pytest (unit testing)
* GitHub Actions (CI automation)

---

# Project Structure

```
cats_dogs_mlops/
│
├── src/
│   ├── model.py              # CNN architecture
│   ├── preprocess.py         # image preprocessing utilities
│
├── app/
│   └── main.py               # FastAPI inference service
│
├── tests/
│   └── test_preprocess.py    # unit tests
│
├── train.py                  # training + MLflow logging
├── requirements.txt
├── Dockerfile
├── docker-compose.yml
└── .github/workflows/ci.yml
```

---

#  Setup Instructions

##  Clone repository

```
git clone <repo-url>
cd cats_dogs_mlops
```

---

## Create virtual environment

### Mac/Linux

```
python -m venv venv
source venv/bin/activate
```

### Windows

```
python -m venv venv
venv\Scripts\activate
```

Install dependencies:

```
pip install -r requirements.txt
```

---

# Dataset Preparation

Download Kaggle dataset:

```
kaggle datasets download -d tongpython/cat-and-dog
unzip cat-and-dog.zip
```

Organize:

```
data/
   train/
      cat/
      dog/
```

Images are:

* resized to **224×224 RGB**
* split into **80/10/10** (train/val/test)

---

# Model Development & Experiment Tracking (M1)

Run training:

```
python train.py
```

This will:

* train CNN model
* save artifact `model.h5`
* log metrics and artifacts in **MLflow**

Launch MLflow UI:

```
mlflow ui
```

Open:

```
http://localhost:5000
```

---

# Run Inference Service Locally (M2)

Start API:

```
uvicorn app.main:app --reload
```

Health check:

```
curl http://localhost:8000/health
```

Prediction:

```
curl -X POST -F file=@cat.jpg http://localhost:8000/predict
```

---

# Docker Containerization

Build image:

```
docker build -t catsdogs .
```

Run container:

```
docker run -p 8000:8000 catsdogs
```

---

# Deployment Using Docker Compose

```
docker-compose up --build
```

Stop:

```
docker-compose down
```

---

# Continuous Integration (M3)

GitHub Actions pipeline automatically:

* checks out code
* installs dependencies
* runs **pytest unit tests**
* builds Docker image

Triggered on:

* push
* pull request

Pipeline file:

```
.github/workflows/ci.yml
```

---

# Continuous Deployment (M4)

Deployment target options:

### ✔ Docker Compose (included)

For local or VM deployment.

### ✔ Kubernetes (optional extension)

Use:

* Deployment YAML
* Service YAML

Apply:

```
kubectl apply -f deployment.yaml
```

---

# Monitoring & Logging (M5)

Implemented:

* API request logging
* prediction logging
* error tracking

Can be extended with:

* Prometheus metrics
* Grafana dashboards
* MLflow Model Registry
* drift detection

---

# Smoke Tests

After deployment:

```
curl http://localhost:8000/health
```

Prediction test:

```
curl -X POST -F file=@test.jpg http://localhost:8000/predict
```

Pipeline should fail if:

* service unavailable
* prediction endpoint errors

---

# Learning Outcomes

This project demonstrates:

* reproducible ML pipelines
* experiment tracking
* model packaging
* containerized inference
* CI/CD automation
* production-style deployment
* monitoring integration

---

# Submission Notes

This repository contains:

* complete source code
* training pipeline
* containerization files
* CI workflow
* tests
* deployment configuration

All components are runnable locally and suitable for academic evaluation.

---

# License

Educational use only.
