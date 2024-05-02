"""
   Preprocess the excel data files to export a new combined training data set

"""

import pandas as pd

# Step 1: Read the dataset
sales2019 = pd.read_excel('SALES2019_test.xlsx')
sales2020 = pd.read_excel('SALES2020_test.xlsx')
forecast2019 = pd.read_excel('BDG2019_test.xlsx')
forecast2020 = pd.read_excel('BDG2020_test.xlsx')

# Delete the first row
forecast2019 = forecast2019.drop(0)
forecast2020 = forecast2020.drop(0)

# Reset index after dropping the row
forecast2019 = forecast2019.reset_index(drop=True)
forecast2020 = forecast2020.reset_index(drop=True)

# Function to add year column
def add_year_column(df, year):
    # Add year column
    df['Έτος'] = year
    return df

# Add year column for sales dataframes
sales2019 = add_year_column(sales2019, 2019)
sales2020 = add_year_column(sales2020, 2020)
forecast2019 = add_year_column(forecast2019, 2019)
forecast2020 = add_year_column(forecast2020, 2020)

print(sales2019)
print(sales2020)
print(forecast2019)
print(forecast2020)

def reshape_data(df, df_type):
    # Check if 'Κωδικός' and 'Περιγραφή' columns are present
    if 'Κωδικός' not in df.columns:
        # If not present, create a new 'Κωδικός' column with index as values
        df['Κωδικός'] = df.index + 1  # Increment index to start from 1
    if 'Περιγραφή' not in df.columns:
        # If not present, create a new 'Περιγραφή' column with empty strings
        df['Περιγραφή'] = ''
    
    # Convert 'Κωδικός' to string with leading zeros
    df['Κωδικός'] = df['Κωδικός'].astype(str).str.zfill(6)
    
    # Determine the value column name
    value_column = 'Sales' if df_type == 'Sales' else 'Forecasts'
    
    # Reshape the dataframe
    reshaped_df = df.melt(id_vars=['Κωδικός', 'Περιγραφή', 'Έτος'], var_name='Μήνας', value_name=value_column)
    
    # Define order of months
    month_order = ['Ιανουάριος', 'Φεβρουάριος', 'Μάρτιος', 'Απρίλιος', 'Μάιος', 'Ιούνιος', 
                   'Ιούλιος', 'Αύγουστος', 'Σεπτέμβριος', 'Οκτώβριος', 'Νοέμβριος', 'Δεκέμβριος']
    
    # Convert 'Μήνας' column to categorical with defined order
    reshaped_df['Μήνας'] = pd.Categorical(reshaped_df['Μήνας'], categories=month_order, ordered=True)
    
    # Sort the dataframe by Product code, Year, and Month
    reshaped_df = reshaped_df.sort_values(by=['Κωδικός', 'Έτος', 'Μήνας'])
    
    return reshaped_df


# Reshape the dataframes
sales2019_reshaped = reshape_data(sales2019, 'Sales')
sales2020_reshaped = reshape_data(sales2020, 'Sales')
forecast2019_reshaped = reshape_data(forecast2019, 'Forecast')
forecast2020_reshaped = reshape_data(forecast2020, 'Forecast')

# Concatenate the reshaped dataframes
combined_sales = pd.concat([sales2019_reshaped, sales2020_reshaped])
combined_forecast = pd.concat([forecast2019_reshaped, forecast2020_reshaped])

# Sort the dataframe by custom product order, year, and month
combined_sales = combined_sales.sort_values(by=['Κωδικός', 'Μήνας', 'Έτος',], ascending=[True, True, True])
combined_forecast = combined_forecast.sort_values(by=['Κωδικός', 'Μήνας', 'Έτος',], ascending=[True, True, True])

# Merge the combined sales and forecast dataframes
combined_data = pd.merge(combined_sales, combined_forecast, on=['Κωδικός', 'Έτος', 'Περιγραφή','Μήνας'])


# Print the resulting table
print("SALES and FORECAST for ALL years:")
print(combined_data)