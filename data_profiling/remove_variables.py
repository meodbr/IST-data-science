import pandas as pd

# Define the columns you want to remove
columns_to_remove = [
    "x1", "x4", "x5", "x6", "x7", "x8", "x9", "x12", "x14", "x16", "x17", "x18", "x19", "x20",
    "x21", "x22", "x23", "x25", "x26", "x27", "x28", "x29", "x31", "x32", "x33", "x34", "x35", 
    "x36", "x37", "x38", "x39", "x41", "x42", "x43", "x44", "x45", "x46", "x47", "x48", "x49", 
    "x52", "x53", "x54", "x57", "x58", "x59", "x69", "x81", "x82"
]

# Load the original dataset
filename = "dataset/class_financial distress.csv"  # Replace with your actual file path
data = pd.read_csv(filename)

# Drop the specified columns
data_cleaned = data.drop(columns=columns_to_remove)

# Cut the dataset to the first 100 rows
data_cleaned = data_cleaned.head(100)

# Save the cleaned dataset to a new CSV file
output_filename = "dataset/cleaned_financial_distresss.csv"  # Replace with your desired output file path
data_cleaned.to_csv(output_filename, index=False)

print(f"Cleaned dataset saved as {output_filename}")
