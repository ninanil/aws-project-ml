# Breast Cancer Prediction Using AWS SageMaker and XGBoost

This project implements a **breast cancer prediction system** using the **Breast Cancer Wisconsin Dataset**, trained and deployed on **AWS SageMaker** using **XGBoost**. The model predicts whether a tumor is malignant or benign based on various medical measurements.

---

## **Features**
- Load and preprocess the breast cancer dataset.
- Train/test split for model evaluation.
- Upload training and test data to Amazon S3 for use in SageMaker.
- Train an XGBoost model on SageMaker with customized hyperparameters.
- Deploy the trained model to a SageMaker endpoint for real-time inference.

---

## **Technologies Used**
- **Python**: Data processing and integration.
- **AWS SageMaker**: Model training and deployment.
- **XGBoost**: Machine learning model for binary classification.
- **Amazon S3**: Storage for training and test data.

---

## **Setup and Usage**

### **Prerequisites**
1. AWS Account with SageMaker and S3 access.
2. Python 3.6 or later installed.
3. Required Python packages:
   - `boto3`
   - `sagemaker`
   - `pandas`
   - `scikit-learn`
4. An existing S3 bucket (e.g., `sagemaker-build-and-deploy-model-sagemaker`).


