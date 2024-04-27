"""
   Evaluate the given model on the test data and print classification report, 
   confusion matrix, and training score.

"""
import pandas as pd
import numpy as np
from statsmodels.tsa.arima.model import ARIMA

def load_sales_forecasts(file_path):
    # Load sales forecasts from Excel file
    sales_forecasts = pd.read_excel(file_path, index_col=0)
    return sales_forecasts

def load_actual_sales(file_path):
    # Load actual sales from Excel file
    actual_sales = pd.read_excel(file_path, index_col=0)
    return actual_sales

def fit_arima_model(train_data):
    # Define ARIMA model parameters
    p = 3  # AutoRegressive (AR) term
    d = 1  # Differencing (I) term
    q = 2  # Moving Average (MA) term

    # Instantiate and fit the ARIMA model
    model = ARIMA(train_data, order=(p, d, q))
    fitted_model = model.fit()
    return fitted_model

def generate_adjusted_forecasts(fitted_model, current_year_forecasts):
    # Forecast future sales
    forecast_steps = current_year_forecasts.shape[1]  # Number of forecasted months
    forecast, stderr, conf_int = fitted_model.forecast(steps=forecast_steps)
   
    # Adjust forecasts based on historical ratio
    historical_ratio = current_year_forecasts / fitted_model.fittedvalues[-len(current_year_forecasts):]
    adjusted_forecasts = forecast * historical_ratio.mean()
   
    return adjusted_forecasts

def main():
    # Load historical sales forecasts and actual sales data
    historical_sales_forecasts = load_sales_forecasts("historical_sales_forecasts.xlsx")
    historical_actual_sales = load_actual_sales("historical_actual_sales.xlsx")

    # Load sales forecasts for the current year
    current_year_forecasts = load_sales_forecasts("current_year_forecasts.xlsx")

    # Train ARIMA model with historical actual sales data
    fitted_model = fit_arima_model(historical_actual_sales)

    # Generate adjusted forecasts for the current year
    adjusted_forecasts = generate_adjusted_forecasts(fitted_model, current_year_forecasts)

    # Create a DataFrame for adjusted forecasts
    adjusted_forecasts_df = pd.DataFrame(adjusted_forecasts, index=current_year_forecasts.index, columns=current_year_forecasts.columns)

    # Save adjusted forecasts to Excel file
    adjusted_forecasts_df.to_excel("adjusted_forecasts.xlsx")

if __name__ == "__main__":
    main()