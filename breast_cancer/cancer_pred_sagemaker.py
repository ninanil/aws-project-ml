# Import necessary libraries
import pandas as pd  # For data manipulation
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
import boto3
import sagemaker
from sagemaker import get_execution_role
from sagemaker.image_uris import retrieve  # Updated for retrieving container images

# Initialize SageMaker session and role
sagemaker_session = sagemaker.Session()
role = get_execution_role()

# Define S3 bucket name (ensure this bucket exists and you have access)
bucket_name = 'sagemaker-build-and-deploy-model-sagemaker'

# Define S3 output location for the model artifacts
s3_output_location = f's3://{bucket_name}/output/'

# Load the breast cancer dataset
data = load_breast_cancer()
X, y = data.data, data.target

# Convert to pandas DataFrame for easier manipulation
df = pd.DataFrame(X, columns=data.feature_names)
df['target'] = y

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Create DataFrames for training and testing
train_data = pd.DataFrame(X_train, columns=data.feature_names)
train_data['target'] = y_train

test_data = pd.DataFrame(X_test, columns=data.feature_names)
test_data['target'] = y_test

# Save training data to CSV
train_csv_path = 'train_data.csv'
train_data.to_csv(train_csv_path, header=False, index=False)

# Save testing data to CSV
test_csv_path = 'test_data.csv'
test_data.to_csv(test_csv_path, header=False, index=False)

# Initialize Boto3 S3 resource
s3 = boto3.resource('s3')

# Upload training data to S3
train_key = 'data/train/train_data.csv'
s3.Bucket(bucket_name).Object(train_key).upload_file(train_csv_path)
print(f"Training data uploaded to s3://{bucket_name}/{train_key}")

# Upload testing data to S3
test_key = 'data/test/test_data.csv'
s3.Bucket(bucket_name).Object(test_key).upload_file(test_csv_path)
print(f"Testing data uploaded to s3://{bucket_name}/{test_key}")

# Retrieve the XGBoost container image URI
xgboost_image = retrieve(
    framework='xgboost',
    region=boto3.Session().region_name,
    version='1.5-1'  # Specify the XGBoost version as needed
)

# Create an XGBoost Estimator
xgb_estimator = sagemaker.estimator.Estimator(
    image_uri=xgboost_image,
    role=role,
    instance_count=1,
    instance_type='ml.m4.xlarge',
    volume_size=5,  # in GB
    output_path=s3_output_location,
    sagemaker_session=sagemaker_session
)

# Set hyperparameters for binary classification
xgb_estimator.set_hyperparameters(
    max_depth=5,
    eta=0.2,
    gamma=4,
    min_child_weight=6,
    objective='binary:logistic',  # Binary classification objective
    num_round=50,
    verbosity=1  # Changed from 'silent' to 'verbosity' for newer XGBoost versions
)

# Define S3 paths for training and testing data
train_s3_path = f's3://{bucket_name}/data/train/'
test_s3_path = f's3://{bucket_name}/data/test/'

# Define data channels using SageMaker's TrainingInput
from sagemaker.inputs import TrainingInput

train_input = TrainingInput(
    s3_data=train_s3_path,
    content_type='csv'
)

test_input = TrainingInput(
    s3_data=test_s3_path,
    content_type='csv'
)

data_channels = {
    'train': train_input,
    'test': test_input  

# Train the model
print("Starting model training...")
xgb_estimator.fit(inputs=data_channels)
print("Model training completed.")

# Deploy the trained model to an endpoint
print("Deploying the model...")
xgb_predictor = xgb_estimator.deploy(
    initial_instance_count=1,
    instance_type='ml.m4.xlarge'
)
print("Model deployed.")

# Save the endpoint name for future use
endpoint_name = xgb_predictor.endpoint_name
print(f"Endpoint name: {endpoint_name}")
