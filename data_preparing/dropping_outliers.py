from pandas import read_csv, DataFrame, Series
from dslabs_functions import (
    NR_STDEV,
    get_variable_types,
    determine_outlier_thresholds_for_var,
)

# Update the file path to the correct location of your dataset
file_path = "../dataset/classification/class_financial_distress.csv"

# Load the dataset
data: DataFrame = read_csv(file_path, na_values="", parse_dates=True)

# Drop missing values
data.dropna(inplace=True)
print(f"Data after dropping missing values: {data.shape}")

# Handling outliers
n_std: int = NR_STDEV
numeric_vars: list[str] = get_variable_types(data)["numeric"]

if numeric_vars is not None:
    df: DataFrame = data.copy(deep=True)
    summary: DataFrame = data[numeric_vars].describe()
    for var in numeric_vars:
        # Determine the outlier thresholds for each numeric variable
        top_threshold, bottom_threshold = determine_outlier_thresholds_for_var(
            summary[var]
        )
        # Identify outliers
        outliers: Series = df[(df[var] > top_threshold) | (df[var] < bottom_threshold)]
        # Drop outliers
        df.drop(outliers.index, axis=0, inplace=True)
    
    # Save the cleaned dataset to a new CSV file
    df.to_csv("class_financial_distress_drop_outliers.csv", index=False)
    print(f"Data after dropping outliers: {df.shape}")
else:
    print("There are no numeric variables")
