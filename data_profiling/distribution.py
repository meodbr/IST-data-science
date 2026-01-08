import os
from numpy import log
from pandas import DataFrame, Series, read_csv
from scipy.stats import norm, expon, lognorm
from matplotlib import pyplot as plt
from matplotlib.axes import Axes

# Constants
HEIGHT = 4  # Height for each subplot (adjust based on your needs)
file_tag = "cleaned_financial_distress"  # Replace with your dataset's tag or name

# Create 'images' directory if it doesn't exist
os.makedirs("images", exist_ok=True)

# Load the dataset
filename = "dataset/cleaned_financial_distresss.csv"  # Replace with your actual file path
data: DataFrame = read_csv(filename, na_values="", index_col=None)

# Select numeric variables from the dataset
numeric = data.select_dtypes(include=["number"]).columns.tolist()

# Define function to compute distributions for a variable
def compute_known_distributions(x_values: list) -> dict:
    distributions = dict()
    
    # Fit Gaussian distribution (Normal)
    mean, sigma = norm.fit(x_values)
    distributions[f"Normal(%.1f, %.2f)" % (mean, sigma)] = norm.pdf(x_values, mean, sigma)
    
    # Fit Exponential distribution
    loc, scale = expon.fit(x_values)
    distributions[f"Exp(%.2f)" % (1 / scale)] = expon.pdf(x_values, loc, scale)
    
    # Fit Log-normal distribution
    sigma, loc, scale = lognorm.fit(x_values)
    distributions[f"LogNor(%.1f, %.2f)" % (log(scale), sigma)] = lognorm.pdf(x_values, sigma, loc, scale)
    
    return distributions

# Define function to plot histogram and fit distributions
def histogram_with_distributions(ax: Axes, series: Series, var: str):
    # Sort the values and convert to list
    values = series.sort_values().to_list()
    
    # Plot histogram
    ax.hist(values, bins=20, density=True, alpha=0.6, color="skyblue", edgecolor="black")
    
    # Get distributions
    distributions = compute_known_distributions(values)
    
    # Plot the fitted distributions
    for label, distribution in distributions.items():
        ax.plot(values, distribution, label=label)
    
    # Set labels and title
    ax.legend(loc="best")
    ax.set_title(f"Best fit for {var}")
    ax.set_xlabel(var)
    ax.set_ylabel("Density")

# Check if there are numeric variables to plot
if numeric:
    # Calculate number of rows and columns for subplots
    rows = len(numeric) // 3 + (1 if len(numeric) % 3 != 0 else 0)  # Adjust the number of rows based on the number of variables
    cols = 3  # You can adjust the number of columns per row

    # Create subplots
    fig, axs = plt.subplots(rows, cols, figsize=(cols * HEIGHT, rows * HEIGHT), squeeze=False)

    # Loop through numeric variables and create histograms with distributions
    i, j = 0, 0
    for n in range(len(numeric)):
        histogram_with_distributions(axs[i, j], data[numeric[n]].dropna(), numeric[n])
        i, j = (i + 1, 0) if (n + 1) % cols == 0 else (i, j + 1)

    # Save and display the plot
    plt.tight_layout()
    plt.savefig(f"images/{file_tag}_histogram_numeric_distribution.png")
    plt.show()

else:
    print("There are no numeric variables.")
