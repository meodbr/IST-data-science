from pandas import DataFrame, read_csv

def mvi_by_dropping(
    data: DataFrame, min_pct_per_var: float = 0.1, min_pct_per_rec: float = 0.0
) -> DataFrame:
    """
    Removes variables and records based on the percentage of valid (non-missing) values.

    Parameters:
    - data: Input DataFrame.
    - min_pct_per_var: Minimum percentage of non-missing values required for a variable to be retained.
    - min_pct_per_rec: Minimum percentage of non-missing values required for a record to be retained.

    Returns:
    - DataFrame with missing variables and records dropped.
    """
    # Step 1: Delete variables (columns) based on the threshold
    df = data.dropna(
        axis=1,  # Drop columns
        thresh=data.shape[0] * min_pct_per_var,  # Minimum valid entries required
        inplace=False,  # Do not modify the original dataset
    )

    # Step 2: Delete records (rows) based on the threshold
    df.dropna(
        axis=0,  # Drop rows
        thresh=df.shape[1] * min_pct_per_rec,  # Minimum valid entries required
        inplace=True,  # Modify the dataset in place
    )

    return df

# Main function
if __name__ == "__main__":
    # Load your dataset
    file_path = "dataset/classification/encoded_set_1.csv"  # Replace with the path to your CSV file
    data: DataFrame = read_csv(file_path)

    # Define thresholds
    min_pct_per_variable = 0.7  # Keep variables with at least 70% valid values
    min_pct_per_record = 0.9    # Keep records with at least 90% valid values

    # Apply missing value removal and name the cleaned dataset as 'cleaned_set_1'
    cleaned_set_1 = mvi_by_dropping(data, min_pct_per_var=min_pct_per_variable, min_pct_per_rec=min_pct_per_record)

    # Output the shape of the cleaned dataset
    print(f"Original dataset shape: {data.shape}")
    print(f"Cleaned dataset shape: {cleaned_set_1.shape}")

    # Save the cleaned dataset to a new CSV file
    cleaned_set_1.to_csv("dataset/classification/no_mv_set_1.csv", index=False)
