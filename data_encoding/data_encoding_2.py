import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from numpy import ndarray
from pandas import DataFrame, concat
from pathlib import Path

def ordinal_encode_column(df: DataFrame, column: str, encoding_map: dict = None) -> DataFrame:
    """
    Performs ordinal encoding for a specific column.
    If no encoding map is provided, assigns an ordinal value to each unique value in the column.

    Args:
        df (pd.DataFrame): The dataset.
        column (str): The column to encode.
        encoding_map (dict, optional): A predefined mapping for encoding. Defaults to None.

    Returns:
        pd.DataFrame: The dataframe with the column encoded.
    """
    if encoding_map:
        df[column] = (
            df[column]
            .fillna(-1)  # Fill NaN with a numeric placeholder value
            .replace(encoding_map)
            .astype(int)  # Safely cast to integer
        )
    else:
        unique_values = df[column].fillna("missing").astype(str).unique()
        encoding_map = {value: idx for idx, value in enumerate(sorted(unique_values))}
        df[column] = (
            df[column]
            .fillna("missing")
            .astype(str)
            .replace(encoding_map)
            .astype(int)  # Safely cast to integer
        )
    return df

def dummify(df: DataFrame, vars_to_dummify: list[str]) -> DataFrame:
    other_vars = [col for col in df.columns if col not in vars_to_dummify]
    enc = OneHotEncoder(
        handle_unknown="ignore", sparse_output=False, dtype="bool", drop=None
    )
    trans: ndarray = enc.fit_transform(df[vars_to_dummify])
    new_vars = enc.get_feature_names_out(vars_to_dummify)
    dummy = pd.DataFrame(trans, columns=new_vars, index=df.index)
    final_df = concat([df[other_vars], dummy], axis=1)
    return final_df

def main(input_csv_path: str, output_csv_path: str):
    data = pd.read_csv(input_csv_path)
    columns_to_drop = ["ARREST_KEY", "PD_DESC", "KY_CD", "OFNS_DESC"]
    data = data.drop(columns=columns_to_drop)
    data = ordinal_encode_column(data, "PD_CD")
    perp_sex_map = {"M": 0, "F": 1}
    age_group_map = {"UNKNOWN":0, "<18": 1, "18-24": 2, "25-44": 3, "45-64": 4, "65+": 5}
    data = ordinal_encode_column(data, "PERP_SEX", perp_sex_map)
    data = ordinal_encode_column(data, "AGE_GROUP", age_group_map)
    data = ordinal_encode_column(data, "ARREST_BORO")
    data = ordinal_encode_column(data, "LAW_CODE")
    vars_to_dummify = ["PERP_RACE"]
    data = dummify(data, vars_to_dummify)
    data.to_csv(output_csv_path, index=False)
    print(f"Processed dataset saved to {output_csv_path}")

if __name__ == "__main__":
    input_csv = Path("../dataset/class_ny_arrests.csv")
    output_csv = Path("../dataset/encoding_v2_set_1.csv")
    main(input_csv, output_csv)
