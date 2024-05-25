from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
import pickle
from sklearn.preprocessing import OneHotEncoder, StandardScaler
import uvicorn

app = FastAPI()

# Load models, encoder, and scaler
with open("../data/logistic_regression.pkl", "rb") as f:
    logistic_regression_model = pickle.load(f)

with open("../data/random_forest.pkl", "rb") as f:
    random_forest_model = pickle.load(f)

with open("../data/gradient_boosting.pkl", "rb") as f:
    gradient_boosting_model = pickle.load(f)

with open("../data/encoder.pkl", "rb") as f:
    encoder = pickle.load(f)

with open("../data/scaler.pkl", "rb") as f:
    scaler = pickle.load(f)

# Define request and response models
class VehicleData(BaseModel):
    Mileage: int
    Age: int
    Number_of_Repairs: int
    Type_of_Vehicle: str
    Days_Since_Last_Maintenance: int

class PredictionResponse(BaseModel):
    Logistic_Regression: int
    Random_Forest: int
    Gradient_Boosting: int

# Preprocess input data
def preprocess_input(data: VehicleData):
    df = pd.DataFrame([data.dict()])
    encoded_features = encoder.transform(df[["Type_of_Vehicle"]])
    encoded_df = pd.DataFrame(encoded_features.toarray(), columns=encoder.get_feature_names_out(["Type_of_Vehicle"]))
    df = pd.concat([df, encoded_df], axis=1)
    df.drop("Type_of_Vehicle", axis=1, inplace=True)
    
    df[["Mileage", "Age", "Days_Since_Last_Maintenance", "Number_of_Repairs"]] = scaler.transform(df[["Mileage", "Age", "Days_Since_Last_Maintenance", "Number_of_Repairs"]])
    
    # Ensure all columns are present
    for col in encoder.get_feature_names_out(["Type_of_Vehicle"]):
        if col not in df.columns:
            df[col] = 0
    
    return df

# Prediction endpoint
@app.post("/predict", response_model=PredictionResponse)
def predict(data: VehicleData):
    df = preprocess_input(data)
    
    logistic_pred = logistic_regression_model.predict(df)[0]
    random_forest_pred = random_forest_model.predict(df)[0]
    gradient_boosting_pred = gradient_boosting_model.predict(df)[0]
    
    return PredictionResponse(
        Logistic_Regression=logistic_pred,
        Random_Forest=random_forest_pred,
        Gradient_Boosting=gradient_boosting_pred
    )

# Root endpoint
@app.get("/")
def read_root():
    return {"message": "Welcome to the Vehicle Maintenance Prediction API"}


if __name__ == "__main__":
    uvicorn.run(app, host="localhost", port=8000)