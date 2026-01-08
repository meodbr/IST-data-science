# Import necessary libraries
from pandas import read_csv, DataFrame
from matplotlib.pyplot import figure, savefig, show
import os

# Create 'images' directory if it doesn't exist
os.makedirs("images", exist_ok=True)

# File details
filename = "dataset/class_financial distress.csv"
file_tag = "financial_distress"

# Load the dataset
data: DataFrame = read_csv(filename, na_values="", index_col=None)  # Adjust index_col if needed

# Analyze missing values
mv: dict[str, int] = {}
for var in data.columns:
    nr: int = data[var].isna().sum()  # Count missing values in the column
    if nr > 0:
        mv[var] = nr

# Display missing values
if mv:
    print("Missing values per variable:")
    for var, count in mv.items():
        print(f"  {var}: {count}")
else:
    print("No missing values found in the dataset.")

# Bar chart function
def plot_bar_chart(categories, values, title="", xlabel="", ylabel=""):
    import matplotlib.pyplot as plt
    plt.bar(categories, values, color="orange", edgecolor="black")
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.xticks(rotation=45, ha="right")  # Rotate x-axis labels for readability

# Plot and save the chart
if mv:
    figure(figsize=(6, 4))
    plot_bar_chart(
        list(mv.keys()),
        list(mv.values()),
        title="Nr of missing values per variable",
        xlabel="variables",
        ylabel="nr missing values"
    )
    savefig(f"images/{file_tag}_mv.png")
    show()
else:
    print("No missing values to plot.")
