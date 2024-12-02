from numpy import ndarray
from pandas import DataFrame, read_csv
from sklearn.model_selection import train_test_split
from matplotlib.pyplot import savefig, show, figure
from dslabs_functions import plot_multibar_chart, evaluate_approach
from balance import balance_dataset

def split_and_clean_data(data: DataFrame, target: str = "class", test_size: float = 0.3) -> tuple:
    # Split the dataset into train and test (70% training, 30% testing)
    train, test = train_test_split(data, test_size=test_size, random_state=42)
    
    return train, test

def main(dataset_path: str, target: str = "class", balancing_method: str = "nothing", divide_by = 1) -> None:
    # Load the dataset
    data: DataFrame = read_csv(dataset_path)
    # Vérifiez les valeurs uniques dans la colonne cible
    # Vérifiez les valeurs uniques dans la colonne cible
    # Split and clean the data (70% training, 30% testing)
    train, test = split_and_clean_data(data, target=target, test_size=0.3)

    # Balance the dataset
    if balancing_method != "nothing":
        train = balance_dataset(train, balancing_method, target=target)

    # Decimate the dataset
    if divide_by != 1:
        fraction = 1/divide_by
        print(f"Taking a fraction of the dataset: {fraction}")
        train = train.sample(frac=fraction, random_state=42)
    
    # Evaluate the approach
    print("Evaluating the approach...")
    eval_results: dict[str, list] = evaluate_approach(train, test, target=target, metric="accuracy")
    
    # Plot and save the evaluation results
    figure()
    plot_multibar_chart(
        ["NB", "KNN"], eval_results, title="Set 1 scaling zscore evaluation", percentage=True
    )
    savefig("./images/set_1_evaluation_scaling_zscore.png")
    show()


# Example of running the main function with your dataset
balancing_method = "nothing"  # Change this to "nothing", "undersampling", "oversampling" or "SMOTE" if you want to use another method

divide_by = 100  # Change this to 1 if you don't want to decimate the dataset
dataset_path = "../dataset/classification/encoded_set_1_replacing_outliers_scaled_zscore.csv"  # Change this path to your actual dataset
target = "JURISDICTION_CODE"

# divide_by = 1  # Change this to 1 if you don't want to decimate the dataset
# dataset_path = "../dataset/classification/set2_replacing_outliers.csv"  # Change this path to your actual dataset
# target = "CLASS"

main(dataset_path, target, balancing_method, divide_by)  # "CLASS" should be the target variable of your dataset
