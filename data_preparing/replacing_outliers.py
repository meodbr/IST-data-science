from pandas import read_csv, DataFrame
from dslabs_functions import (
    get_variable_types,
    determine_outlier_thresholds_for_var,
)

# Update the file path to the correct location of your dataset
file_path = "../dataset/classification/encoded_set_1_filled_mv.csv"

# Load the dataset
data: DataFrame = read_csv(file_path, na_values="", parse_dates=True)

# Identify numeric variables
numeric_vars = get_variable_types(data)["numeric"]

if numeric_vars:
    # Create a deep copy of the dataset to avoid modifying the original
    df: DataFrame = data.copy(deep=True)

    # Process numeric variables
    for var in numeric_vars:
        # Determine thresholds for outliers
        summary = df[numeric_vars].describe()
        top_threshold, bottom_threshold = determine_outlier_thresholds_for_var(summary[var])

        # Compute the median of the column
        median: float = df[var].median()

        # Replace outliers with the median value
        df[var] = df[var].apply(lambda x: median if x > top_threshold or x < bottom_threshold else x)

    # Replace missing values with the median
    for var in numeric_vars:
        median: float = df[var].median()
        df[var].fillna(median, inplace=True)

    # Save the cleaned dataset to a new CSV file
    df.to_csv("../dataset/classification/encoded_set_1_replacing_outliers.csv", index=False)
    print(f"Data after dropping outliers: {df.shape}")
else:
    print("There are no numeric variables in the dataset.")
