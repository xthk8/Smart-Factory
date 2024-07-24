import pandas as pd

# Load the data
data_path = "C:/Users/USER/Desktop/설비학회/Dataset_장비이상 조기탐지 AI 데이터셋/data/5공정_180sec/total_date.csv"  # Update with your file path
data = pd.read_csv(data_path)

# Ensure 'date' column is in datetime format
data['Date'] = pd.to_datetime(data['Date'])

# Sort the data by 'process' and 'date'
sorted_data = data.sort_values(by=['Process', 'Date'])

# Save the sorted data to a new CSV file
sorted_data.to_csv("C:/Users/USER/Desktop/설비학회/Dataset_장비이상 조기탐지 AI 데이터셋/data/5공정_180sec/lot_seq.csv", index=False)
