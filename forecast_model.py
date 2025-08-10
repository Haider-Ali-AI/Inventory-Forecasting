import pandas as pd
from prophet import Prophet
import matplotlib.pyplot as plt

print("1. Script started.")

# 1. Load and prepare the data
print("2. Attempting to load data from CSV files...")
try:
    train_df = pd.read_csv('train.csv')
    holidays_df = pd.read_csv('holidays_events.csv')
    print("3. Data loaded successfully.")
except FileNotFoundError:
    print("Error: Please download the 'train.csv' and 'holidays_events.csv' files from the Kaggle competition and place them in the project folder.")
    exit()

# We'll use the 'family' column to select a product category.
# The 'DAIRY' family is the one you identified.
print("4. Filtering for 'store_nbr' == 1 and 'family' == 'DAIRY'...")
SKU_to_forecast = train_df[(train_df['store_nbr'] == 1) & (train_df['family'] == 'DAIRY')].copy()

print(f"   Found {len(SKU_to_forecast)} rows for this combination.")

# Clean and preprocess the data for Prophet
# Prophet requires 'ds' (date) and 'y' (target variable) columns.
SKU_to_forecast['date'] = pd.to_datetime(SKU_to_forecast['date'])
SKU_to_forecast = SKU_to_forecast.rename(columns={'date': 'ds', 'sales': 'y'})

# Remove any rows with missing or invalid data, as Prophet cannot train on them.
SKU_to_forecast.dropna(inplace=True)

print(f"   Found {len(SKU_to_forecast)} valid rows after cleaning.")

# Check if there are enough rows to train the model
if len(SKU_to_forecast) < 2:
    print("\n-------------------------------------------")
    print("Error: Not enough valid data to train the model.")
    print("The selected combination of 'store_nbr' and 'family' has too many missing 'sales' or 'date' values.")
    print("Please try a different combination, such as one of the most common families:")
    print(train_df['family'].value_counts().head(5))
    print("-------------------------------------------")
    exit()

# Prepare the holidays data for Prophet
holidays_df['date'] = pd.to_datetime(holidays_df['date'])
holidays_df = holidays_df.rename(columns={'date': 'ds', 'description': 'holiday'})
national_holidays = holidays_df[holidays_df['locale'] == 'National']

print("5. Data preparation complete.")

# 2. Train the Prophet model
print("6. Attempting to train the Prophet model...")
try:
    model = Prophet(holidays=national_holidays, yearly_seasonality=True, weekly_seasonality=True)
    model.fit(SKU_to_forecast)
    print("7. Model training complete.")
except Exception as e:
    print(f"An error occurred during model training: {e}")
    exit()

# 3. Create a future DataFrame and forecast
print("8. Attempting to forecast sales...")
try:
    future = model.make_future_dataframe(periods=30, freq='D')
    forecast = model.predict(future)
    print("9. Forecasting complete.")
except Exception as e:
    print(f"An error occurred during forecasting: {e}")
    exit()

# 4. Output the results
print("10. Displaying results...")
try:
    print("\n-------------------------------------------")
    print("Predicted Sales for the Next 30 Days (Excerpt):")
    print("-------------------------------------------")
    print(forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail())
    
    # 5. Visualize the forecast
    print("11. Generating forecast plot...")
    fig1 = model.plot(forecast)
    plt.title("Sales Forecast with Prophet")
    plt.xlabel("Date")
    plt.ylabel("Sales Quantity")
    
    print("12. Generating component plot...")
    fig2 = model.plot_components(forecast)
    plt.show()
    print("13. Script finished successfully.")
except Exception as e:
    print(f"An error occurred while plotting or displaying results: {e}")
    exit()