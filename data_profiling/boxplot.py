# Import necessary libraries
from pandas import read_csv, DataFrame
from matplotlib.pyplot import savefig, show, subplots
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

if numeric:
    # Define grid dimensions based on the number of numeric variables
    def define_grid(n: int) -> tuple[int, int]:
        from math import ceil, sqrt
        rows = ceil(sqrt(n))
        cols = ceil(n / rows)
        return rows, cols

    HEIGHT = 4  # Adjust figure height per subplot
    rows, cols = define_grid(len(numeric))

    # Create subplots for boxplots
    fig, axs = subplots(rows, cols, figsize=(cols * HEIGHT, rows * HEIGHT), squeeze=False)

    # Plot boxplots for each numeric variable
    i, j = 0, 0
    for n in range(len(numeric)):
        axs[i, j].set_title(f"Boxplot for {numeric[n]}")
        axs[i, j].boxplot(data[numeric[n]].dropna().values)
        i, j = (i + 1, 0) if (n + 1) % cols == 0 else (i, j + 1)

    # Hide unused subplots if any
    for remaining_ax in axs.flatten()[len(numeric):]:
        remaining_ax.set_visible(False)

    # Save and show the plot
    savefig(f"images/{file_tag}_numeric_boxplots.png")
    show()
else:
    print("There are no numeric variables.")
