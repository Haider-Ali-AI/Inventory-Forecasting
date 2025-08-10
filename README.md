Store Sales Forecasting Application
This project provides a simple, yet powerful, web application for forecasting future store sales using a pre-trained machine learning model. The application is built with Streamlit for the user interface and uses Facebook's Prophet library for time series forecasting.

Project Workflow
The project is structured into two main parts to ensure a clean and efficient workflow:

Model Training: A Python script (train_and_save_model.py) trains a Prophet model on historical data and saves the trained model and other necessary data (like holiday information) to disk using joblib and pandas.to_pickle. This step is a one-time process and is separate from the application's runtime.

Web Application: A Streamlit application (streamlit_app_v2.py) loads the pre-trained model and data, eliminating the need to retrain the model every time the app is run. Users can interact with a simple interface to specify a forecasting period and visualize the results.

Technologies Used
Python: The core programming language for the project.

Prophet: An open-source library developed by Facebook for forecasting time series data.

Streamlit: A framework for rapidly creating and sharing custom web apps for machine learning and data science.

Pandas: A data manipulation and analysis library, used for handling time series data and saving/loading dataframes.

Joblib: A library for saving and loading Python objects, used here to efficiently serialize the Prophet model.

Matplotlib: Used by Prophet to generate the forecasting plots.

How to Run the Application
To get this application up and running, follow these two simple steps:

Train and Save the Model:
First, run the train_and_save_model.py script. This will create two files in your directory: prophet_model.joblib and holidays.pkl.

python train_and_save_model.py

Run the Streamlit App:
After the model files have been created, you can launch the interactive web application using Streamlit.

streamlit run streamlit_app_v2.py

The application will open in your default web browser, where you can generate and view forecasts.
