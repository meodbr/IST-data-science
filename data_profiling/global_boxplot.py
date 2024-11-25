# Import necessary libraries
from pandas import read_csv, DataFrame
from matplotlib.pyplot import figure, savefig, show, boxplot, title, xlabel, ylabel
import os

# Create 'images' directory if it doesn't exist
os.makedirs("images", exist_ok=True)

# File details
filename = "dataset/class_financial distress.csv"  # Replace with your dataset path
file_tag = "financial_distress"  # Identifier for saving output files

# Load the dataset
data: DataFrame = read_csv(filename, na_values="", index_col=None)

# Select numeric variables
numeric = data.select_dtypes(include=["number"]).columns.tolist()

# Create global boxplot
if numeric:
    # Combine all numeric variables into a single series
    combined_data = data[numeric].melt(value_name="value")["value"].dropna()

    # Create a single boxplot
    figure(figsize=(6, 4))
    boxplot(combined_data.values, vert=True, patch_artist=True, boxprops=dict(facecolor="lightblue"))
    title("Global Boxplot for All Numeric Variables")
    xlabel("All Numeric Data")
    ylabel("Values")

    # Save and show the global boxplot
    savefig(f"images/{file_tag}_global_boxplot.png")
    show()
else:
    print("There are no numeric variables.")
