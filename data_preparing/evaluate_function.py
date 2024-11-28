from numpy import ndarray
from pandas import DataFrame, read_csv
from sklearn.model_selection import train_test_split
from matplotlib.pyplot import savefig, show, figure
from dslabs_functions import plot_multibar_chart, CLASS_EVAL_METRICS, run_NB, run_KNN

def evaluate_approach(
    train: DataFrame, test: DataFrame, target: str = "class", metric: str = "accuracy"
) -> dict[str, list]:
    # Separate target variable and features
    trnY = train.pop(target).values
    trnX: ndarray = train.values
    tstY = test.pop(target).values
    tstX: ndarray = test.values
    eval: dict[str, list] = {}

    # Run Naive Bayes and KNN on the data
    eval_NB: dict[str, float] = run_NB(trnX, trnY, tstX, tstY, metric=metric)
    eval_KNN: dict[str, float] = run_KNN(trnX, trnY, tstX, tstY, metric=metric)
    
    # Combine results from both classifiers
    if eval_NB != {} and eval_KNN != {}:
        for met in CLASS_EVAL_METRICS:
            eval[met] = [eval_NB[met], eval_KNN[met]]
    
    return eval

def split_and_clean_data(data: DataFrame, target: str = "class", test_size: float = 0.3) -> tuple:
    # Split the dataset into train and test (70% training, 30% testing)
    train, test = train_test_split(data, test_size=test_size, random_state=42)
    
    # Drop missing values in both train and test datasets
    train_cleaned = train.dropna(how='any')
    test_cleaned = test.dropna(how='any')
    
    return train_cleaned, test_cleaned

def main(dataset_path: str, target: str = "class"):
    # Load the dataset
    data: DataFrame = read_csv(dataset_path)
    
    # Split and clean the data (70% training, 30% testing)
    train, test = split_and_clean_data(data, target=target, test_size=0.3)
    
    # Evaluate the approach
    eval_results: dict[str, list] = evaluate_approach(train, test, target=target, metric="accuracy")
    
    # Plot and save the evaluation results
    figure()
    plot_multibar_chart(
        ["NB", "KNN"], eval_results, title="Evaluation after Dropping Missing Values", percentage=True
    )
    savefig("images/evaluation_after_dropping_missing_values.png")
    show()

# Example of running the main function with your dataset
dataset_path = "../dataset/class_financial distress.csv"  # Change this path to your actual dataset
main(dataset_path, target="CLASS")  # "CLASS" should be the target variable of your dataset
