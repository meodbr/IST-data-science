from pandas import read_csv, DataFrame
from dslabs_functions import (
    get_variable_types,
    determine_outlier_thresholds_for_var,
)

# Update the file path to the correct location of your dataset
file_path = "../dataset/classification/encoded_set_1_with_filled_missing_values.csv"

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

        # Truncate outliers
        df[var] = df[var].apply(
            lambda x: top_threshold if x > top_threshold else bottom_threshold if x < bottom_threshold else x
        )

    # Save the cleaned dataset to a new CSV file
    df.to_csv("../dataset/classification/encoded_set_1_truncate_outliers.csv", index=False)
    print(f"Data after truncating outliers: {df.shape}")
else:
    print("There are no numeric variables in the dataset.")
