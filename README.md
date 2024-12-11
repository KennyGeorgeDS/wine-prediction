# Wine Prediction Model and Deployment

## Overview
This project is a demonstration of a deep learning model that predicts whether a wine is RED or WHITE based on its chemical properties. The model is trained on the [Wine Quality Dataset](https://archive.ics.uci.edu/ml/datasets/wine+quality) from the UCI Machine Learning Repository. The dataset contains 12 features and 1 target variable. The features are chemical properties of the wine, such as pH, alcohol content, and acidity. The target variable is the type of wine, which can be either RED or WHITE. The model is trained using a deep learning architecture with PyTorch. The model is then deployed as a RESTful API using FastAPI. The API is deployed on AWS EKS using Docker and Kubernetes.

### Access Points

[Streamlit Dashboard](http://a67ad9e1fe62a416a951d642dcb4cc27-1424045864.us-east-2.elb.amazonaws.com/) 

[FastAPI Service](http://acd7184e54d314c67844b5cba4797e3e-683884404.us-east-2.elb.amazonaws.com/docs)

### Tools Used
- Python
- PyTorch
- SciKit-Learn
- FastAPI
- Docker
- Kubernetes
- AWS EKS
- Poetry

### Project Features
- Deep Learning Model
    - The model is trained using a deep learning architecture with PyTorch.
    - The model is trained on the Wine Quality Dataset from the UCI Machine Learning Repository.
    - The model predicts whether a wine is RED or WHITE based on its chemical properties.
    - The model is saved as a PyTorch model file (.pth) for deployment.
    - The model is trained using `tool.poetry.scripts` in the `pyproject.toml` to run the training script.
- FastAPI Service
    - The model is deployed as a RESTful API using FastAPI.
    - The API has a single endpoint that accepts a POST request with a JSON payload.
    - The API returns a JSON response with the predicted wine type.
    - The API is deployed on AWS EKS using Docker and Kubernetes.
- Streamlit Dashboard
    - The Streamlit dashboard is a web application that allows users to interact with the model.
    - The dashboard has a form where users can input the chemical properties of a wine.
    - The dashboard displays the predicted wine type based on the input.
    - The dashboard is deployed on AWS EKS using Docker and Kubernetes.
- Docker
    - The model, API, and dashboard are containerized using Docker.
    - The Docker images are built using the Dockerfiles in the `deploy` and `dashboard` directories.
    - The Docker images are pushed to Docker Hub for deployment.
- Kubernetes
    - The API and dashboard are deployed on AWS EKS using Kubernetes.
    - The Kubernetes deployment files are in the `deploy/kubernetes` directory.
    - The API and dashboard are exposed as services using LoadBalancer services.

### Project Structure
```
├── app
│  ├── __pycache__
│  │  └── app.cpython-310.pyc
│  └── app.py
├── dashboard
│  ├── Dockerfile
│  └── wine_dashboard.py
├── data
│  ├── model_training.py
│  ├── red_wine.csv
│  └── white_wine.csv
├── deploy
│  ├── Dockerfile
│  └── kubernetes
│     ├── api-deployment.yaml
│     ├── api-service.yaml
│     ├── dashboard-deployment.yaml
│     └── dashboard-service.yaml
├── model
│  ├── data_eda.ipynb
│  ├── training
│  │  ├── __pycache__
│  │  │  └── model_training.cpython-310.pyc
│  │  ├── data_eda.ipynb
│  │  └── model_training.py
│  └── wine_model.pth
├── poetry.lock
├── pyproject.toml
└── README.md
```
