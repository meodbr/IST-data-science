from pandas import DataFrame, concat, Series
from numpy import ndarray
from imblearn.over_sampling import SMOTE

# truncate the majority class to the size of the minority class
def balance_undersampling(train_sep: DataFrame, most_frequent_class, target: str = "class") -> DataFrame:
    majority_class_truncated = train_sep[most_frequent_class].sample(len(train_sep[1 - most_frequent_class]))
    print(f"Majority class truncated to {len(majority_class_truncated)}")
    return concat([majority_class_truncated, train_sep[1 - most_frequent_class]], axis=0)

# oversample the minority class to the size of the majority class
def balance_oversampling(train_sep: DataFrame, most_frequent_class, target: str = "class") -> DataFrame:
    minority_class_oversampled = train_sep[1 - most_frequent_class].sample(len(train_sep[most_frequent_class]), replace=True)
    print(f"Minority class oversampled to {len(minority_class_oversampled)}")
    return concat([train_sep[most_frequent_class], minority_class_oversampled], axis=0)

def balance_SMOTE(train_base: DataFrame, train_sep: DataFrame, most_frequent_class, target: str = "class") -> DataFrame:
    RANDOM_STATE = 42
    smote = SMOTE(sampling_strategy="minority", random_state=RANDOM_STATE)
    y = train_base.pop(target)
    X = train_base.to_numpy()
    smote_X, smote_y = smote.fit_resample(X, y)
    result = DataFrame(concat([smote_y, DataFrame(smote_X)], axis=1))
    result.columns = [target] + train_base.columns.tolist()
    print(f"SMOTE applied to the minority class, new size: {len(result[result[target] == 1])}")
    print(f"Shape: {result.shape}")
    return result


def balance_dataset(train: DataFrame, method = "undersampling", target: str = "class") -> DataFrame:
    # Separate the dataset into target = 0 et target = 1 (positive and negative classes)
    train_separated = [train[train[target] == 0], train[train[target] == 1]]

    # get the most frequent class
    most_frequent_class = 0 if len(train_separated[0]) > len(train_separated[1]) else 1

    text = "Negative" if most_frequent_class == 0 else "Positive"
    print(f"Most frequent class: {text}")
    print(f"Positives : {len(train_separated[1])}, Negatives : {len(train_separated[0])}")

    # use the method to balance the dataset
    if method == "undersampling":
        train_balanced = balance_undersampling(train_separated, most_frequent_class, target=target)
    elif method == "oversampling":
        train_balanced = balance_oversampling(train_separated, most_frequent_class, target=target)
    elif method == "SMOTE":
        train_balanced = balance_SMOTE(train, train_separated, most_frequent_class, target=target)
    else:
        raise ValueError("The method should be either 'undersampling' or 'oversampling'")
    return train_balanced
    