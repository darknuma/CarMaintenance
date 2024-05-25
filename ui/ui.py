import streamlit as st
import requests
import json

# create a streamlit app 
def main ():
    st.title("Car Maintenance Prediction")
    # create a form and buttons 
    Mileage = st.number_input("Mileage the car travels", min_value=0, max_value=200000, value=50000)
    Age = st.number_input("Age of the car", min_value=0, max_value=30, value=5)
    Number_of_Repairs = st.number_input("Number of Repairs Done", min_value=0, max_value=20, value=1)
    
    Type_of_Car = st.selectbox(
        "Select the type of vehicle",
        ("Sedan", "SUV", "Truck", "Convertible")
    )

    Days_Since_Last_Maintenance = st.number_input("How many days since your last car maintenance", min_value=0, max_value=365, value=100)

    # Prepare input data for the API
    input_data = {
        "Mileage": Mileage,
        "Age": Age,
        "Number_of_Repairs": Number_of_Repairs,
        "Type_of_Vehicle": Type_of_Car,
        "Days_Since_Last_Maintenance": Days_Since_Last_Maintenance
    }

    if st.button("Predict"):
        try:
            response = requests.post("http://localhost:8000/predict", json=input_data)
            if response.status_code == 200:
                predictions = response.json()
                logistic_pred = predictions["Logistic_Regression"]
                random_forest_pred = predictions["Random_Forest"]
                gradient_boosting_pred = predictions["Gradient_Boosting"]

                st.write("## Prediction Results:")
                st.write(f"Logistic Regression: {'Maintenance Needed' if logistic_pred == 1 else 'No Maintenance Needed'}")
                st.write(f"Random Forest: {'Maintenance Needed' if random_forest_pred == 1 else 'No Maintenance Needed'}")
                st.write(f"Gradient Boosting: {'Maintenance Needed' if gradient_boosting_pred == 1 else 'No Maintenance Needed'}")
            else:
                st.error("Error in API response")
        except Exception as e:
            st.error(f"Error in API request: {e}")

if __name__ == "__main__":
    main()