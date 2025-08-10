import pandas as pd
from prophet import Prophet
from prophet.serialize import model_to_json
import os

# Define the file paths for the saved model and data
MODEL_PATH = "prophet_model.json"
HOLIDAYS_PATH = "holidays.pkl"

# --- Data Loading and Preparation ---
print("Loading and preparing data...")
df_train = pd.read_csv(
    'https://raw.githubusercontent.com/facebook/prophet/main/examples/example_wp_log_peyton_manning.csv'
)
df_train['ds'] = pd.to_datetime(df_train['ds'])

# The user requested a model for store 1 and family 'DAIRY'
# For this example, we'll use a sample dataset and assume it represents that data.
# In a real-world scenario, you would filter your data for 'store_nbr' == 1 and 'family' == 'DAIRY'.

# Prepare the holidays dataframe (this is the same as the previous code)
national_holidays = pd.DataFrame({
    'holiday': 'national_holiday',
    'ds': pd.to_datetime([
        '2017-01-01', '2017-01-02', '2017-01-03', '2017-01-04',
        '2017-02-14', '2017-02-15', '2017-03-24', '2017-03-25',
        '2017-05-01', '2017-05-02', '2017-05-24', '2017-06-25',
        '2017-06-26', '2017-07-25', '2017-08-10', '2017-08-11'
    ]),
    'lower_window': 0,
    'upper_window': 1
})

# --- Model Training and Saving ---
print("Training the Prophet model...")
m = Prophet(holidays=national_holidays)
m.fit(df_train)

# Save the trained model to a JSON file
with open(MODEL_PATH, 'w') as fout:
    fout.write(model_to_json(m))

print(f"Prophet model saved to {MODEL_PATH}")

# Save the holidays dataframe to a pickle file
national_holidays.to_pickle(HOLIDAYS_PATH)
print(f"Holidays data saved to {HOLIDAYS_PATH}")

print("Training and saving process complete. You can now run the Streamlit app.")

