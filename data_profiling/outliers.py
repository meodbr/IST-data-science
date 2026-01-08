# Import necessary libraries
from pandas import DataFrame, Series, read_csv
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure, savefig, show
import os

# Constants for standard deviation and IQR thresholds
NR_STDEV: int = 2
IQR_FACTOR: float = 1.5

# Create 'images' directory if it doesn't exist
os.makedirs("images", exist_ok=True)

# Define function to determine outlier thresholds based on standard deviation or IQR
def determine_outlier_thresholds_for_var(
    summary5: Series, std_based: bool = True, threshold: float = NR_STDEV
) -> tuple[float, float]:
    """
    Determines the top and bottom outlier thresholds for a given variable
    using either standard deviation or IQR-based method.
    """
    top, bottom = 0, 0
    if std_based:
        # Standard deviation-based method
        std = threshold * summary5["std"]
        top = summary5["mean"] + std
        bottom = summary5["mean"] - std
    else:
        # IQR-based method
        iqr = threshold * (summary5["75%"] - summary5["25%"])
        top = summary5["75%"] + iqr
        bottom = summary5["25%"] - iqr
    return top, bottom

# Define function to count outliers in the dataset
def count_outliers(
    data: DataFrame,
    numeric: list[str],
    nrstdev: int = NR_STDEV,
    iqrfactor: float = IQR_FACTOR,
) -> dict:
    """
    Count outliers using both standard deviation and IQR methods.
    Returns a dictionary with counts for each method.
    """
    outliers_iqr = []
    outliers_stdev = []
    summary5 = data[numeric].describe()

    # Loop through numeric variables and calculate outliers
    for var in numeric:
        # Standard deviation-based outliers
        top, bottom = determine_outlier_thresholds_for_var(
            summary5[var], std_based=True, threshold=nrstdev
        )
        outliers_stdev.append(
            data[data[var] > top].count()[var] + data[data[var] < bottom].count()[var]
        )

        # IQR-based outliers
        top, bottom = determine_outlier_thresholds_for_var(
            summary5[var], std_based=False, threshold=iqrfactor
        )
        outliers_iqr.append(
            data[data[var] > top].count()[var] + data[data[var] < bottom].count()[var]
        )

    return {"iqr": outliers_iqr, "stdev": outliers_stdev}

# Load the dataset
filename = "dataset/class_financial distress.csv"  # Update with your file path
file_tag = "financial_distress"  # Identifier for the output file

# Load the dataset
data: DataFrame = read_csv(filename, na_values="", index_col=None)

# Select numeric variables
numeric = data.select_dtypes(include=["number"]).columns.tolist()

# If there are numeric variables, count the outliers
if numeric:
    outliers = count_outliers(data, numeric)

    # Create a bar chart for the outlier counts using matplotlib
    figure(figsize=(12, 6))

    # Plot standard deviation-based outliers
    plt.bar(numeric, outliers["stdev"], width=0.4, label="Standard Deviation", align="center", color="skyblue")

    # Plot IQR-based outliers
    plt.bar(numeric, outliers["iqr"], width=0.4, label="IQR", align="edge", color="orange")

    plt.title("Number of Outliers per Variable (Standard Deviation vs IQR)")
    plt.xlabel("Variables")
    plt.ylabel("Number of Outliers")
    plt.xticks(rotation=45, ha="right")  # Rotate x-axis labels for readability
    plt.legend()

    # Save and display the plot
    savefig(f"images/{file_tag}_outliers.png")
    show()
else:
    print("There are no numeric variables.")
