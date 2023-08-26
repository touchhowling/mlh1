from fastapi import FastAPI
from pydantic import BaseModel
from typing import Optional
import uvicorn

app = FastAPI()

class TestInput(BaseModel):
    P_NAME: Optional[str]
    P_STATUS: int
    P_MASS: float
    P_MASS_ERROR_MIN: float
    P_MASS_ERROR_MAX: float
    P_RADIUS: Optional[str]
    P_RADIUS_ERROR_MIN: Optional[str]
    P_RADIUS_ERROR_MAX: Optional[str]
    P_SEMI_MAJOR_AXIS_EST: float
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, LabelEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline

# Load data from CSV file
data = pd.read_csv("phl_exoplanet_catalog_2019.csv")  # Replace with your CSV file

# Convert string values to numeric using LabelEncoder
label_encoder = LabelEncoder()
for column in data.columns:
    if data[column].dtype == "object":
        data[column] = label_encoder.fit_transform(data[column])

# Separate target variable and features
y = data["P_HABITABLE"].values
X = data.drop(columns=["P_HABITABLE"])

# Set up column transformations using a pipeline
preprocessor = ColumnTransformer(
    transformers=[
        ('num', SimpleImputer(strategy='median'), X.columns)
    ])

# Create a pipeline with preprocessing and model
pipeline = Pipeline(steps=[('preprocessor', preprocessor),
                           ('model', RandomForestClassifier())])

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the model
pipeline.fit(X_train, y_train)

# Predict habitability for test data
predicted_habitability = pipeline.predict(X_test)

# Calculate accuracy
accuracy = (predicted_habitability == y_test).mean()
print("Accuracy:", accuracy)

# Testing function
def test_model(example):
    example_df = pd.DataFrame([example])  # Wrap the example data in a DataFrame
    example_df = example_df.apply(lambda col: label_encoder.transform(col) if col.name in label_encoder.classes_ else col)
    example_pred = pipeline.predict(example_df)
    if example_pred[0] == 1:
        return "Habitable"
    else:
        return "Not Habitable"

from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI()


@app.get("/")
async def process_data(input_data:dict):
    result = test_model(input_data)  # Assuming test_model is defined somewhere
    return {"result": result}
#uvicorn.run(app)
