# Importing necessary libraries
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

# Missing values analysis
mv = {}
for var in data.columns:
    missing_count = data[var].isna().sum()
    if missing_count > 0:
        mv[var] = missing_count

# Plot missing values
if mv:
    figure(figsize=(6, 4))
    plot_bar_chart(
        list(mv.keys()),
        list(mv.values()),
        title="Nr of missing values per variable",
        xlabel="Variables",
        ylabel="Nr missing values",
    )
    savefig(f"images/{file_tag}_missing_values.png")
    show()
else:
    print("No missing values in the dataset.")

# Dataset dimensionality summary
print(f"Number of records: {nr_records}")
print(f"Number of variables: {nr_variables}")
print("Missing values per variable:")
for var, count in mv.items():
    print(f"  {var}: {count}")
