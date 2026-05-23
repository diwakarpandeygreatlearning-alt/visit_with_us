# for data manipulation
import pandas as pd
import sklearn
# for creating a folder
import os
# for data preprocessing and pipeline creation
from sklearn.model_selection import train_test_split
# for hugging face space authentication to upload files
from huggingface_hub import login, HfApi

# Define constants for the dataset and output paths
api = HfApi(token=os.getenv("HF_TOKEN"))
DATASET_PATH = "hf://datasets/diwakarpandey-greatlearning/trip-prediction/tourism.csv"
tourism_dataset = pd.read_csv(DATASET_PATH)
print("Dataset loaded successfully.")

# Fix inconsistent values in Gender column
# Replace 'Fe male' with 'Female'
tourism_dataset['Gender'] = tourism_dataset['Gender'].replace('Fe male', 'Female')
tourism_dataset['Gender'] = tourism_dataset['Gender'].str.strip().str.title()

# Replace 'Unmarried' with 'Single'
tourism_dataset['MaritalStatus'] = tourism_dataset['MaritalStatus'].replace('Unmarried', 'Single')
tourism_dataset['MaritalStatus'] = tourism_dataset['MaritalStatus'].str.strip().str.title()


# Define the target variable for the classification task
target = 'ProdTaken'

# List of numerical features in the dataset
numeric_features = [
    'Age',                       # Age of the customer
    'CityTier',                  # City category (Tier 1 > Tier 2 > Tier 3)
    'DurationOfPitch',           # Duration of the sales pitch delivered
    'NumberOfPersonVisiting',    # Number of people accompanying the customer
    'NumberOfFollowups',         # Total number of follow-ups after the pitch
    'PreferredPropertyStar',     # Preferred hotel rating (e.g., 3-star, 5-star)
    'NumberOfTrips',             # Average number of trips taken annually
    'Passport',                  # Whether the customer holds a valid passport (0/1)
    'PitchSatisfactionScore',    # Satisfaction score for the sales pitch
    'OwnCar',                    # Whether the customer owns a car (0/1)
    'NumberOfChildrenVisiting',  # Number of children below age 5 accompanying
    'MonthlyIncome'              # Gross monthly income of the customer
]

# List of categorical features in the dataset
categorical_features = [
    'TypeofContact',             # Method of contact (Company Invited / Self Inquiry)
    'Occupation',                # Customer's occupation (Salaried, Freelancer, etc.)
    'Gender',                    # Gender of the customer (Male/Female)
    'ProductPitched',            # Type of product pitched to the customer
    'MaritalStatus',             # Marital status (Single, Married, Divorced)
    'Designation'                # Customer's designation in their organization
]
# Define predictor matrix (X) using selected numeric and categorical features
X = tourism_dataset[numeric_features + categorical_features]

# Define target variable
y = tourism_dataset[target]

# Split dataset into train and test
Xtrain, Xtest, ytrain, ytest = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42
)

# Save splits locally
Xtrain.to_csv("Xtrain.csv", index=False)
Xtest.to_csv("Xtest.csv", index=False)
ytrain.to_csv("ytrain.csv", index=False)
ytest.to_csv("ytest.csv", index=False)

# Upload files to Hugging Face dataset repo
files = ["Xtrain.csv", "Xtest.csv", "ytrain.csv", "ytest.csv"]

for file_path in files:
    api.upload_file(
        path_or_fileobj=file_path,
        path_in_repo=file_path.split("/")[-1],  # just the filename
        repo_id="diwakarpandey-greatlearning/trip-prediction",
        repo_type="dataset",
    )
