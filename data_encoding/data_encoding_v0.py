import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from numpy import ndarray
from pandas import DataFrame, concat
from pathlib import Path

def ordinal_encode_column(df: DataFrame, column: str) -> DataFrame:
    """
    Automatically performs ordinal encoding for a specific column.
    Assigns an ordinal value to each unique value in the column.

    Args:
        df (pd.DataFrame): The dataset.
        column (str): The column to encode.

    Returns:
        pd.DataFrame: The dataframe with the column encoded.
    """
    unique_values = df[column].unique()
    encoding_map = {value: idx for idx, value in enumerate(sorted(unique_values))}
    df[column] = df[column].replace(encoding_map)
    return df

def dummify(df: DataFrame, vars_to_dummify: list[str]) -> DataFrame:
    """
    Applies dummification (One-Hot Encoding) to the specified variables in the DataFrame.

    Args:
        df (pd.DataFrame): The dataset.
        vars_to_dummify (list[str]): List of variables to dummify.

    Returns:
        pd.DataFrame: A new DataFrame with dummified variables.
    """
    # Variables not to be dummified
    other_vars = [col for col in df.columns if col not in vars_to_dummify]

    # Apply OneHotEncoder
    enc = OneHotEncoder(
        handle_unknown="ignore", sparse_output=False, dtype="bool", drop=None
    )
    trans: ndarray = enc.fit_transform(df[vars_to_dummify])

    # Get the names of the new dummy variables
    new_vars = enc.get_feature_names_out(vars_to_dummify)

    # Create a DataFrame with the dummy variables
    dummy = pd.DataFrame(trans, columns=new_vars, index=df.index)

    # Concatenate the original non-dummified variables with the dummy variables
    final_df = concat([df[other_vars], dummy], axis=1)

    return final_df

def main(input_csv_path: str, output_csv_path: str):
    """
    Preprocess the dataset by dropping specified columns, performing ordinal encoding on PD_CD,
    dummifying PERP_RACE, and saving the processed dataset.

    Args:
        input_csv_path (str): Path to the input dataset.
        output_csv_path (str): Path to save the processed dataset.
    """
    # Load the dataset
    data = pd.read_csv(input_csv_path)

    # Drop specified columns
    columns_to_drop = ["ARREST_KEY", "PD_DESC", "KY_CD"]
    data = data.drop(columns=columns_to_drop)

    # Perform ordinal encoding on the PD_CD column
    data = ordinal_encode_column(data, "PD_CD")

    # Dummify the PERP_RACE variable
    vars_to_dummify = ["PERP_RACE"]
    data = dummify(data, vars_to_dummify)

    # Save the processed dataset
    data.to_csv(output_csv_path, index=False)
    print(f"Processed dataset saved to {output_csv_path}")

# Example usage
if __name__ == "__main__":
    # Define paths
    input_csv = Path("../dataset/class_ny_arrests.csv")
    output_csv = Path("../dataset/first_encoding_set_1.csv")

    # Run the main function
    main(input_csv, output_csv)