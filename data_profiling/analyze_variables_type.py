# Import necessary libraries
from pandas import read_csv, DataFrame, Series, to_numeric, to_datetime
from matplotlib.pyplot import figure, savefig, show
import os

# Create 'images' directory if it doesn't exist
os.makedirs("images", exist_ok=True)

# File details
filename = "dataset/class_ny_arrests.csv"
file_tag = "ny_arrests"

# Load the dataset
data: DataFrame = read_csv(filename, na_values="", index_col=None)  # Adjust index_col if needed

# Define a function to determine variable types
def get_variable_types(df: DataFrame) -> dict[str, list]:
    """
    Classify variables into types: numeric, binary, symbolic, and date.
    """
    variable_types: dict[str, list] = {"numeric": [], "binary": [], "date": [], "symbolic": []}
    nr_values: Series = df.nunique(axis=0, dropna=True)  # Count unique values for each column
    
    for c in df.columns:
        if nr_values[c] == 2:  # Binary variables
            variable_types["binary"].append(c)
            df[c] = df[c].astype("bool")
        else:
            try:
                # Try converting to numeric
                to_numeric(df[c], errors="raise")
                variable_types["numeric"].append(c)
            except ValueError:
                try:
                    # Try converting to datetime
                    df[c] = to_datetime(df[c], errors="raise")
                    variable_types["date"].append(c)
                except ValueError:
                    # Default to symbolic
                    variable_types["symbolic"].append(c)
    
    return variable_types

# Analyze variable types
variable_types = get_variable_types(data)

# Display variable types
print("Variable types:")
for vtype, variables in variable_types.items():
    print(f"  {vtype}: {variables}")

# Count the number of variables for each type
counts = {vtype: len(variables) for vtype, variables in variable_types.items()}

# Bar chart function
def plot_bar_chart(categories, values, title="", xlabel="", ylabel=""):
    import matplotlib.pyplot as plt
    plt.bar(categories, values, color="skyblue", edgecolor="black")
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)

# Plot and save the bar chart
figure(figsize=(4, 2))
plot_bar_chart(
    list(counts.keys()),
    list(counts.values()),
    title="Number of variables per type",
    xlabel="Variable type",
    ylabel="Count"
)
savefig(f"images/{file_tag}_variable_types.png")
show()

# Transform symbolic variables to category
symbolic_vars = variable_types["symbolic"]
if symbolic_vars:
    data[symbolic_vars] = data[symbolic_vars].apply(lambda x: x.astype("category"))

# Print updated data types
print("\nUpdated data types:")
print(data.dtypes)
