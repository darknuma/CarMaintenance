import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import classification_report
import pickle

# Generate a simulated dataset
def generate_data(num_records):
    np.random.seed(42)
    data = {
        "Vehicle_ID": range(num_records),
        "Mileage": np.random.randint(10000, 200000, num_records),
        "Age": np.random.randint(1, 15, num_records),
        "Number_of_Repairs": np.random.randint(0, 10, num_records),
        "Type_of_Vehicle": np.random.choice(["Sedan", "SUV", "Truck", "Convertible"], num_records),
        "Last Maintenance Date": (pd.Timestamp.today() - pd.to_timedelta(np.random.randint(1, 365, num_records), unit="D")),
        "Maintenance_Needed": np.random.randint(0, 2, num_records),
    }
    return pd.DataFrame(data)

# Simulate data for 1000 records
df = generate_data(1000)

# Data Preprocessing
def preprocess_data(df, encoder=None, scaler=None, fit=False):
    df["Days_Since_Last_Maintenance"] = (pd.Timestamp.today() - df["Last Maintenance Date"]).dt.days
    df.drop("Last Maintenance Date", axis=1, inplace=True)
    df.drop("Vehicle_ID", axis=1, inplace=True)
    
    if fit:
        encoder = OneHotEncoder()
        encoded_features = encoder.fit_transform(df[["Type_of_Vehicle"]])
        scaler = StandardScaler()
        df[["Mileage", "Age", "Days_Since_Last_Maintenance", "Number_of_Repairs"]] = scaler.fit_transform(df[["Mileage", "Age", "Days_Since_Last_Maintenance", "Number_of_Repairs"]])
    else:
        encoded_features = encoder.transform(df[["Type_of_Vehicle"]])
        df[["Mileage", "Age", "Days_Since_Last_Maintenance", "Number_of_Repairs"]] = scaler.transform(df[["Mileage", "Age", "Days_Since_Last_Maintenance", "Number_of_Repairs"]])
    
    encoded_df = pd.DataFrame(encoded_features.toarray(), columns=encoder.get_feature_names_out(["Type_of_Vehicle"]))
    df = pd.concat([df, encoded_df], axis=1)
    df.drop("Type_of_Vehicle", axis=1, inplace=True)
    
    return df, encoder, scaler

# Preprocess the dataframe
processed_df, encoder, scaler = preprocess_data(df, fit=True)

# Splitting data into training and testing sets
X = processed_df.drop("Maintenance_Needed", axis=1)
y = processed_df["Maintenance_Needed"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train and save models
def train_save_model(model, model_name):
    model.fit(X_train, y_train)
    with open(f"../data/{model_name}.pkl", "wb") as f:
        pickle.dump(model, f)

train_save_model(LogisticRegression(), "logistic_regression")
train_save_model(RandomForestClassifier(), "random_forest")
train_save_model(GradientBoostingClassifier(), "gradient_boosting")

# Save the encoder and scaler
with open("../data/encoder.pkl", "wb") as f:
    pickle.dump(encoder, f)

with open("../data/scaler.pkl", "wb") as f:
    pickle.dump(scaler, f)

# Print classification reports for validation
def print_classification_report(model_name):
    with open(f"../data/{model_name}.pkl", "rb") as f:
        model = pickle.load(f)
    y_pred = model.predict(X_test)
    print(f"Classification report for {model_name}:\n", classification_report(y_test, y_pred))

print_classification_report("logistic_regression")
print_classification_report("random_forest")
print_classification_report("gradient_boosting")
