
from pandas import read_csv, DataFrame, Series
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from matplotlib.pyplot import subplots, show

# MinMax_Scaler
chemin = "/home/mina/Documents/portugal/dataScience/class_financial_distress"
data: DataFrame = read_csv(chemin+".csv", na_values="")
target = "CLASS"

vars: list[str] = data.columns.to_list()
vars.remove(target)  # On enlève la colonne cible de la liste des variables

target_data: Series = data.pop(target)

transf: MinMaxScaler = MinMaxScaler(feature_range=(0, 1), copy=True).fit(data)
df_minmax = DataFrame(transf.transform(data))
df_minmax[target] = target_data
df_minmax.columns = vars + [target]
df_minmax.to_csv(chemin+"_scaled_minmax.csv", index=False)

# Normalisation des données avec StandardScaler
transf: StandardScaler = StandardScaler(with_mean=True, with_std=True, copy=True).fit(data)
# Transformation des données pour avoir une distribution standard (moyenne 0, écart-type 1)
df_zscore = DataFrame(transf.transform(data))
# Ajouter la variable cible normalisée dans le DataFrame
df_zscore[target] = target_data
df_zscore.columns = vars + [target]
df_zscore.to_csv(chemin+"_scaled_zscore.csv", index=False) 


# Afficher
fig, axs = subplots(1, 3, figsize=(20, 10), squeeze=False)
axs[0, 1].set_title("Original data")
data.boxplot(ax=axs[0, 0])
axs[0, 0].set_title("Z-score normalization")
df_zscore.boxplot(ax=axs[0, 1])
axs[0, 2].set_title("MinMax normalization")
df_minmax.boxplot(ax=axs[0, 2])

fig.savefig('boxplot_comparison.png')

