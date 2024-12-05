import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from numpy import ndarray
from pandas import DataFrame, concat
from pathlib import Path
from math import sin, cos, pi
import pandas as pd
import numpy as np



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
        df[column] = df[column].map(encoding_map)

    else:
        df[column] = pd.Categorical(df[column]).codes
    # Remplace les valeurs -1 (catégories manquantes dans pd.Categorical) par NaN
    df[column] = df[column].replace(-1, float("nan"))
    return df

def dummify(df: DataFrame, vars_to_dummify: list[str]) -> DataFrame:
    dummy = pd.get_dummies(
        df[vars_to_dummify], 
        prefix=vars_to_dummify, 
        dtype="int"
    )
    other_vars = [col for col in df.columns if col not in vars_to_dummify]
    return pd.concat([df[other_vars], dummy], axis=1)


def encode_date_features(data: pd.DataFrame, date_col: str, reference_date: str = "01/01/2006") -> pd.DataFrame:
    """
    Encode les informations temporelles d'une colonne de date :
    - Jour de la semaine (cyclique)
    - Jour du mois (cyclique)
    - Mois de l'année (cyclique)
    - Nombre de jours écoulés depuis une date de référence

    Args:
        data (pd.DataFrame): DataFrame contenant la colonne de date.
        date_col (str): Nom de la colonne contenant les dates au format 'dd/mm/yyyy'.
        reference_date (str): Date de référence au format 'dd/mm/yyyy' (par défaut : 01/01/2006).
    
    Returns:
        pd.DataFrame: DataFrame avec les nouvelles colonnes ajoutées.
    """
    data[date_col] = pd.to_datetime(data[date_col], format="%m/%d/%Y", errors="coerce")
    ref_date = pd.to_datetime(reference_date, format="%m/%d/%Y")
    invalid_dates = data[date_col].isna().sum()
    if invalid_dates > 0:
        print(f"{invalid_dates} valeurs invalides dans la colonne '{date_col}' ont été transformées en NaT.")

    # Calculer toutes les informations temporelles avec Numpy
    day_of_week = data[date_col].dt.weekday + 1
    day_of_month = data[date_col].dt.day
    month_of_year = data[date_col].dt.month
    days_since_ref = (data[date_col] - ref_date).dt.days

    data["day_of_week_sin"] = np.sin(2 * np.pi * day_of_week / 7)
    data["day_of_week_cos"] = np.cos(2 * np.pi * day_of_week / 7)
    data["day_of_month_sin"] = np.sin(2 * np.pi * day_of_month / 31)
    data["day_of_month_cos"] = np.cos(2 * np.pi * day_of_month / 31)
    data["month_of_year_sin"] = np.sin(2 * np.pi * month_of_year / 12)
    data["month_of_year_cos"] = np.cos(2 * np.pi * month_of_year / 12)
    data["days_since_reference"] = days_since_ref

    return data


def main(input_csv_path: str, output_csv_path: str):
    data = pd.read_csv(input_csv_path)

    # Encode 'JURISDICTION_CODE' preserving missing values
    data["JURISDICTION_CODE"] = data["JURISDICTION_CODE"].apply(
        lambda x: 1.0 if pd.notna(x) and x <= 2 else (0.0 if pd.notna(x) else np.nan)
    )

    
    data = encode_date_features(data, "ARREST_DATE")
    columns_to_drop = ["ARREST_KEY", "PD_DESC", "KY_CD", "OFNS_DESC", "ARREST_DATE"]
    data = data.drop(columns=columns_to_drop)

    # Ordinal encoding
    data = ordinal_encode_column(data, "PD_CD")
    perp_sex_map = {"M": 0, "F": 1}
    age_group_map = {"UNKNOWN":0, "<18": 1, "18-24": 2, "25-44": 3, "45-64": 4, "65+": 5}
    law_cat_cd_map = {"M": 0, "F": 1}
    data = ordinal_encode_column(data, "PERP_SEX", perp_sex_map)
    data = ordinal_encode_column(data, "AGE_GROUP", age_group_map)
    data = ordinal_encode_column(data, "ARREST_BORO")
    data = ordinal_encode_column(data, "LAW_CODE")
    data = ordinal_encode_column(data, "LAW_CAT_CD", law_cat_cd_map)

    # One-hot encoding for categorical variable
    vars_to_dummify = ["PERP_RACE"]
    data = dummify(data, vars_to_dummify)
    
    data.to_csv(output_csv_path, index=False)
    print(f"Processed dataset saved to {output_csv_path}")

if __name__ == "__main__":
    input_csv = Path("../dataset/classification/class_ny_arrests.csv")
    output_csv = Path("../dataset/classification/encoded_set_1.csv")
    main(input_csv, output_csv)
