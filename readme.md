# Car Maintenance ML Model with Backend API
I was inspired to do this side project after reading (Balwant Gorad)[https://medium.com/@goradbj/how-to-build-complete-end-to-end-ml-model-backend-restapi-using-fastapi-and-front-end-ui-using-22f64bf04476] medium article, where his focus was to build complete end-to-end ML model, Backend RestAPI using FastAPI and front-end UI using Streamlit. His data set included a Car Price Prediction. 

My goal was to do it differently.

## Hot to Run this Project
* Set up the virtual environment 
 ```sh
    python -m venv venv
```
* Install the requirements 
 ```sh
    pip install -r requirements.txt
```
* change directory to src `cd src`
* Run the python file to:
   * It generates the dataset
   * preprocess the model
   * trains and test the model 
   * dumps the model, scaler and encoder into the `data` folder.
```sh
    python train.py 
```
* Cd to `cd api` to run the backend Rest.api
* cd to `cd ui` to run the Streamlit frontend.



