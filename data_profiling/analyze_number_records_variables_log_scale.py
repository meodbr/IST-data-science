# Importing necessary libraries
from pandas import read_csv, DataFrame
from matplotlib.pyplot import figure, savefig, show
import os

# Create 'images' directory if it doesn't exist
os.makedirs("images", exist_ok=True)

# File details
filename = "dataset/class_ny_arrests.csv"
file_tag = "ny_arrests"

# Load the dataset
data: DataFrame = read_csv(filename, na_values="", index_col=None)  # Adjust index_col if needed

# Analyze the shape of the dataset
print(f"Dataset shape: {data.shape}")
nr_records, nr_variables = data.shape

# Bar chart function with optional log scale or normalization
def plot_bar_chart(categories, values, title="", xlabel="", ylabel="", use_log_scale=False, normalize=False):
    import matplotlib.pyplot as plt
    import numpy as np
    
    if normalize:
        # Normalize the data (e.g., divide nr_records by 1,000,000 for better visualization)
        max_value = max(values)
        values = [v / max_value for v in values]
        ylabel = "Normalized Values"
    
    plt.bar(categories, values, color="skyblue", edgecolor="black")
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    
    if use_log_scale:
        plt.yscale("log")  # Apply logarithmic scale
        ylabel += " (log scale)"
    
    # Annotate values for clarity
    for i, v in enumerate(values):
        plt.text(i, v + (0.05 if normalize else 1), f"{v:.2e}" if use_log_scale else str(v), ha="center")

# Plot and save the bar chart with different approaches
figure(figsize=(6, 4))

# Option 1: Logarithmic scale
plot_bar_chart(
    ["nr records", "nr variables"],
    [nr_records, nr_variables],
    title="Nr of records vs nr variables (Log Scale)",
    xlabel="Category",
    ylabel="Count",
    use_log_scale=True
)

# Option 2: Normalized values
# plot_bar_chart(
#    ["nr records", "nr variables"],
#    [nr_records, nr_variables],
#    title="Nr of records vs nr variables (Normalized)",
#    xlabel="Category",
#    ylabel="Count",
#    normalize=True
#)

# Save and display
savefig(f"images/{file_tag}_records_variables.png")
show()

# Dataset dimensionality summary
print(f"Number of records: {nr_records}")
print(f"Number of variables: {nr_variables}")
