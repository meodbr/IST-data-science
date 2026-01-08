from sklearn.model_selection import train_test_split
from pandas import DataFrame, read_csv


dataset_path = "/home/mina/Documents/portugal/dataScience/balanced_SMOTE_set_2.csv"
data= read_csv(dataset_path)
# Drop column: 'Financial Distress'
data = data.drop(columns=['Company'])
train, test = train_test_split(data, test_size=0.3, random_state=42)

print("Saving dataset...")
train.to_csv("train_dataset_2.csv", index=False)
test.to_csv("test_dataset_2.csv", index=False)
print("c'est tout good ma poule")