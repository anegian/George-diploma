"""
   Preprocess the excel data files to export a new combined training data set

"""

import pandas as pd

def read_xlsx(file_name):
    """
        reading xlsx files and print them with their name
    """
    dataframe = pd.read_excel(file_name)
    print("Reading file: ", file_name, "\n", dataframe)
    return dataframe

def export_data(dataframe, file_name):
    """
        export the modified files to new csv files, in the same directory
    """
    # Export to CSV
    csv_file = file_name + '.csv'
    # Use utf-8-sig for Excel compatibility
    dataframe.to_csv(csv_file, index=False, encoding='utf-8-sig') 
    print("Exported reshaped data to CSV file:", csv_file)

# MAIN STEPS #

# Step 1: Read and print the dataset files
training_data = read_xlsx('training_data.xlsx')

# Step 2: Export the csv file
export_data(training_data, 'training_data')
