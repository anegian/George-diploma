"""
   Preprocess the excel data files to export a new combined training data set

"""

import pandas as pd

def read_and_print_excel(file_name):
    """
        reading xlsx files and print them with their name
    """
    dataframe = pd.read_excel(file_name)
    print("Reading file: ", file_name, "\n", dataframe)
    return dataframe

def reshape_data(dataframe, df_type):
    """
        modify and sort the columns of the tables
    """
    # Define order of months
    month_order = ['Ιανουάριος', 'Φεβρουάριος', 'Μάρτιος', 'Απρίλιος', 'Μάιος', 'Ιούνιος',
                   'Ιούλιος', 'Αύγουστος', 'Σεπτέμβριος', 'Οκτώβριος', 'Νοέμβριος', 'Δεκέμβριος']

    # Determine the value column name
    value_column = 'Sales' if df_type == 'Sales' else 'Forecasts'

    # Melt the DataFrame to reshape it
    reshaped_df = pd.melt(dataframe, id_vars=['Περιγραφή', 'Κωδικός', 'Έτος'], value_vars=month_order, var_name='Μήνας', value_name=value_column)

    # Convert 'Μήνας' column to categorical with defined order
    reshaped_df['Μήνας'] = pd.Categorical(reshaped_df['Μήνας'], categories=month_order, ordered=True)

    # Sort the dataframe by Product code, Year, and Month
    reshaped_df = reshaped_df.sort_values(by=['Κωδικός', 'Μήνας', 'Έτος', ])

    # Convert the Sales or Forecasts column to integers, to avoid decimals
    reshaped_df[value_column] = reshaped_df[value_column].astype(int)

    return reshaped_df

def export_data(dataframe, file_name):
    """
        export the modified files to new xlsx ans csv files, in the same directory
    """
    # Export to Excel
    excel_file = file_name + '.xlsx'
    dataframe.to_excel(excel_file, index=False)
    print("Exported reshaped data to Excel file:", excel_file)

    # Export to CSV
    csv_file = file_name + '.csv'
    # Use utf-8-sig for Excel compatibility
    dataframe.to_csv(csv_file, index=False, encoding='utf-8-sig') 
    print("Exported reshaped data to CSV file:", csv_file)

# MAIN STEPS #

# Step 1: Read and print the dataset files
sales2019 = read_and_print_excel('./SALES/SALES2019.xlsx')
sales2020 = read_and_print_excel('./SALES/SALES2020.xlsx')
sales2021 = read_and_print_excel('./SALES/SALES2021.xlsx')
sales2022 = read_and_print_excel('./SALES/SALES2022.xlsx')
sales2023 = read_and_print_excel('./SALES/SALES2023.xlsx')
forecast2019 = read_and_print_excel('./BDG/BDG2019.xlsx')
forecast2020 = read_and_print_excel('./BDG/BDG2020.xlsx')
forecast2021 = read_and_print_excel('./BDG/BDG2021.xlsx')
forecast2022 = read_and_print_excel('./BDG/BDG2022.xlsx')
forecast2023 = read_and_print_excel('./BDG/BDG2023.xlsx')

#  Step 2: Reshape the dataframes
sales2019_reshaped = reshape_data(sales2019, 'Sales')
sales2020_reshaped = reshape_data(sales2020, 'Sales')
sales2021_reshaped = reshape_data(sales2021, 'Sales')
sales2022_reshaped = reshape_data(sales2022, 'Sales')
sales2023_reshaped = reshape_data(sales2023, 'Sales')
forecast2019_reshaped = reshape_data(forecast2019, 'Forecasts')
forecast2020_reshaped = reshape_data(forecast2020, 'Forecasts')
forecast2021_reshaped = reshape_data(forecast2021, 'Forecasts')
forecast2022_reshaped = reshape_data(forecast2022, 'Forecasts')
forecast2023_reshaped = reshape_data(forecast2023, 'Forecasts')

#  Step 3: Concatenate the reshaped dataframes
combined_sales = pd.concat([sales2019_reshaped, sales2020_reshaped, sales2021_reshaped, sales2022_reshaped, sales2023_reshaped])
combined_forecast = pd.concat([forecast2019_reshaped, forecast2020_reshaped, forecast2021_reshaped, forecast2022_reshaped, forecast2023_reshaped])

# Step 4: Merge the combined sales and forecast dataframes
combined_data = pd.merge(combined_sales, combined_forecast, on=['Περιγραφή', 'Κωδικός', 'Έτος', 'Μήνας'])

# Step 5: Print and export the overall tables
print("SALES and FORECAST for ALL years:")
print(combined_data)
export_data(combined_data, 'training_data')
