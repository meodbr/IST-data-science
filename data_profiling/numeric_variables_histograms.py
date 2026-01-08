# Import necessary libraries
from pandas import DataFrame, read_csv
from matplotlib.pyplot import figure, subplots, savefig, show
import os

# Create 'images' directory if it doesn't exist
os.makedirs("images", exist_ok=True)

# Constants for grid layout
HEIGHT: int = 4  # Height for each subplot, you can adjust based on your needs

# Define function to set chart labels (you can adapt this from your course example)
def set_chart_labels(ax, title: str, xlabel: str, ylabel: str):
    """
    Helper function to set labels for each subplot.
    """
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)

# Load the dataset
filename = "dataset/cleaned_financial_distresss.csv"  # Replace with your actual file path
file_tag = "financial_distress"  # Identifier for the output file

# Load the dataset
data: DataFrame = read_csv(filename, na_values="", index_col=None)

# Select numeric variables
numeric = data.select_dtypes(include=["number"]).columns.tolist()

# If there are numeric variables, generate histograms
if numeric:
    # Calculate number of rows and columns for the grid layout
    rows = len(numeric) // 3 + (1 if len(numeric) % 3 != 0 else 0)
    cols = 3  # Set number of columns per row (you can adjust this)

    # Create subplots
    fig, axs = subplots(rows, cols, figsize=(cols * HEIGHT, rows * HEIGHT), squeeze=False)

    # Loop through numeric variables and create histograms
    i, j = 0, 0
    for n in range(len(numeric)):
        set_chart_labels(
            axs[i, j],
            title=f"Histogram for {numeric[n]}",
            xlabel=numeric[n],
            ylabel="Nr of records"
        )
        # Plot histogram (auto bins)
        axs[i, j].hist(data[numeric[n]].dropna().values, bins="auto", color="skyblue", edgecolor="black")
        # Update subplot positions
        i, j = (i + 1, 0) if (n + 1) % cols == 0 else (i, j + 1)

    # Save and display the histograms
    savefig(f"images/{file_tag}_single_histograms_numeric.png")
    show()
else:
    print("There are no numeric variables.")
