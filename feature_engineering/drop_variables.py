from pandas import read_csv

train_path = "/home/mina/Documents/portugal/dataScience/train_dataset_2.csv"
test_path = "/home/mina/Documents/portugal/dataScience/test_dataset_2.csv"

train_df = read_csv(train_path)
test_df = read_csv(test_path)

columns_to_drop = ['x1', 'x7', 'x12', 'x15', 'x16', 'x17', 'x19', 'x20', 'x31', 'x32', 'x34', 'x35', 'x52', 'x54', 'x57', 'x59']


train_df = train_df.drop(columns=columns_to_drop)
test_df = test_df.drop(columns=columns_to_drop)


train_df.to_csv("train_df_2_w_dropped_variables.csv")
test_df.to_csv("test_df_2_w_dropped_variables.csv")