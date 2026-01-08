# Importing necessary libraries
from pandas import read_csv, DataFrame
from matplotlib.pyplot import figure, savefig, show
import os

# Create 'images' directory if it doesn't exist
os.makedirs("images", exist_ok=True)

# File details
filename = "dataset/class_ny_arrests.csv"
file_tag = "financial_distress"

# Load the dataset
data: DataFrame = read_csv(filename, na_values="", index_col=None)  # Adjust index_col if needed

# Analyze the shape of the dataset
print(f"Dataset shape: {data.shape}")
nr_records, nr_variables = data.shape

# Plotting number of records vs number of variables
figure(figsize=(4, 2))
values = {"nr records": nr_records, "nr variables": nr_variables}
keys, counts = list(values.keys()), list(values.values())

# Bar chart function
def plot_bar_chart(categories, values, title="", xlabel="", ylabel=""):
    import matplotlib.pyplot as plt
    plt.bar(categories, values, color="skyblue", edgecolor="black")
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)

# Plot and save the bar chart
plot_bar_chart(keys, counts, title="Nr of records vs nr variables")
savefig(f"images/{file_tag}_records_variables.png")
show()

# Dataset dimensionality summary
print(f"Number of records: {nr_records}")
print(f"Number of variables: {nr_variables}")