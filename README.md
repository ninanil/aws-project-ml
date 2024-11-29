# AWS Project ML

## **About**
This repository contains projects focused on implementing **machine learning models** using **AWS services** like SageMaker and S3. Each project demonstrates a specific ML task, from data preprocessing to deployment.


---

## **Projects**
### 1. **Breast Cancer Prediction**
- **Description**:
  Predicts whether a tumor is malignant or benign using the Breast Cancer Wisconsin dataset. The model is trained using **XGBoost** on AWS SageMaker and deployed as an endpoint for real-time predictions.

- **Technologies Used**:
  - **Amazon SageMaker**: For training and deploying machine learning models.
  - **Amazon S3**: For storing training and testing datasets.
  - **XGBoost**: A high-performance machine learning algorithm used for binary classification.
  - **Boto3**: AWS SDK for Python to interact with SageMaker and S3.