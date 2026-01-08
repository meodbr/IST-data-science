from math import sin, cos, pi
import pandas as pd

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
    # Convertir les colonnes en datetime
    data[date_col] = pd.to_datetime(data[date_col], format="%m/%d/%Y")
    ref_date = pd.to_datetime(reference_date, format="%m/%d/%Y")
    
    # Jour de la semaine (1 = Lundi, 7 = Dimanche)
    data["day_of_week"] = data[date_col].dt.weekday + 1  # Lundi = 1
    data["day_of_week_sin"] = data["day_of_week"].apply(lambda x: round(sin(2 * pi * x / 7), 3))
    data["day_of_week_cos"] = data["day_of_week"].apply(lambda x: round(cos(2 * pi * x / 7), 3))

    # Jour du mois
    data["day_of_month"] = data[date_col].dt.day
    data["day_of_month_sin"] = data["day_of_month"].apply(lambda x: round(sin(2 * pi * x / 31), 3))
    data["day_of_month_cos"] = data["day_of_month"].apply(lambda x: round(cos(2 * pi * x / 31), 3))

    # Mois de l'année
    data["month_of_year"] = data[date_col].dt.month
    data["month_of_year_sin"] = data["month_of_year"].apply(lambda x: round(sin(2 * pi * x / 12), 3))
    data["month_of_year_cos"] = data["month_of_year"].apply(lambda x: round(cos(2 * pi * x / 12), 3))

    # Nombre de jours écoulés depuis la date de référence
    data["days_since_reference"] = (data[date_col] - ref_date).dt.days

    return data

# Exemple d'utilisation
data = pd.read_csv('dataset\class_ny_arrests.csv')
data = encode_date_features(data, "ARREST_DATE")
data.to_csv("trucmuche.csv")
print(data)
