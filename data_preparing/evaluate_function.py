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

def main(dataset_path: str, target: str = "class", balancing_method: str = "nothing") -> None:
    # Load the dataset
    data: DataFrame = read_csv(dataset_path)
    
    # Split and clean the data (70% training, 30% testing)
    train, test = split_and_clean_data(data, target=target, test_size=0.3)

    # Balance the dataset
    if balancing_method != "nothing":
        train = balance_dataset(balancing_method, train, target=target)
    
    # Evaluate the approach
    eval_results: dict[str, list] = evaluate_approach(train, test, target=target, metric="accuracy")
    
    # Plot and save the evaluation results
    figure()
    plot_multibar_chart(
        ["NB", "KNN"], eval_results, title="Evaluation after Dropping Outliers", percentage=True
    )
    savefig("images/evaluation_after_dropping_outliers.png")
    show()



# Example of running the main function with your dataset
dataset_path = "../dataset/classification/class_financial_distress_drop_outliers.csv"  # Change this path to your actual dataset
balancing_method = "nothing"  # Change this to "nothing", "undersampling", "oversampling" or "SMOTE" if you want to use another method
main(dataset_path, target="CLASS", balancing_method="nothing")  # "CLASS" should be the target variable of your dataset
