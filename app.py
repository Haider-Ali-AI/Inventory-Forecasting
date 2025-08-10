import streamlit as st
import pandas as pd
from prophet import Prophet
import matplotlib.pyplot as plt

# Set page configuration for a wider layout
st.set_page_config(layout="wide", page_title="Inventory Sales Forecast")

# --- Caching Functions ---
# We use st.cache_data to load the data only once,
# which makes the app much faster for subsequent runs.
@st.cache_data
def load_data():
    """Loads and caches the train and holidays CSV files."""
    try:
        train_df = pd.read_csv('train.csv')
        holidays_df = pd.read_csv('holidays_events.csv')
        return train_df, holidays_df
    except FileNotFoundError:
        st.error("Error: Please make sure 'train.csv' and 'holidays_events.csv' are in the same directory.")
        return None, None

@st.cache_data
def get_unique_values(df, column):
    """Returns unique values from a specified column."""
    return sorted(df[column].unique())

# --- Main App Logic ---
st.title("Inventory Sales Forecasting Application")

# Load the data
train_df, holidays_df = load_data()

# Check if data loaded successfully
if train_df is not None and holidays_df is not None:

    # Create a sidebar for user input
    st.sidebar.header("Select Parameters")

    # Get unique store numbers and product families for the dropdowns
    store_numbers = get_unique_values(train_df, 'store_nbr')
    product_families = get_unique_values(train_df, 'family')

    # Create the selectbox widgets in the sidebar
    selected_store = st.sidebar.selectbox(
        "Choose a Store Number",
        options=store_numbers,
        index=0  # Default to the first store
    )

    selected_family = st.sidebar.selectbox(
        "Choose a Product Family",
        options=product_families,
        index=product_families.index('DAIRY') if 'DAIRY' in product_families else 0 # Default to 'DAIRY'
    )
    
    # Add a button to trigger the forecast generation
    if st.sidebar.button("Generate Forecast", type="primary"):
        
        st.subheader(f"Sales Forecast for Store {selected_store}, Family: {selected_family}")

        # Filter the data based on user selection
        SKU_to_forecast = train_df[(train_df['store_nbr'] == selected_store) & (train_df['family'] == selected_family)].copy()

        # Check for sufficient data
        if len(SKU_to_forecast) < 2:
            st.warning("Not enough valid data to train the model for the selected combination. Please select another combination.")
        else:
            # Prepare the data for Prophet
            SKU_to_forecast['date'] = pd.to_datetime(SKU_to_forecast['date'])
            SKU_to_forecast = SKU_to_forecast.rename(columns={'date': 'ds', 'sales': 'y'})
            SKU_to_forecast.dropna(inplace=True)

            # Prepare the holidays data for Prophet
            holidays_df['date'] = pd.to_datetime(holidays_df['date'])
            holidays_df = holidays_df.rename(columns={'date': 'ds', 'description': 'holiday'})
            national_holidays = holidays_df[holidays_df['locale'] == 'National']

            # --- Model Training and Forecasting ---
            try:
                # Initialize and train the Prophet model
                model = Prophet(holidays=national_holidays, yearly_seasonality=True, weekly_seasonality=True)
                model.fit(SKU_to_forecast)
                
                # Create a future DataFrame for forecasting
                future = model.make_future_dataframe(periods=90, freq='D')
                forecast = model.predict(future)

                # --- Display Results in the Main Panel ---

                # Display the main forecast plot
                st.write("#### Sales Forecast Plot")
                fig1 = model.plot(forecast)
                plt.title(f"Sales Forecast for Store {selected_store}, Family: {selected_family}")
                st.pyplot(fig1)

                # Display the forecast components plot
                st.write("#### Forecast Components (Trend, Weekly, Yearly)")
                fig2 = model.plot_components(forecast)
                st.pyplot(fig2)

                # Display the raw forecast data
                st.write("#### Raw Forecast Data")
                st.dataframe(forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail())
                
            except Exception as e:
                st.error(f"An error occurred during forecasting: {e}")

