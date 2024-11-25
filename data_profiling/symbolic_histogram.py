import pandas as pd
import matplotlib.pyplot as plt
import dslabs_functions as df

def plot_symbolic_histograms(data: pd.DataFrame, symbolic_columns: list, file_tag: str):
    """
    Plots histograms for symbolic and binary variables in the dataset.

    Parameters:
        data (pd.DataFrame): The dataset.
        symbolic_columns (list): List of symbolic or binary columns.
        file_tag (str): Tag for saving the output image file.
    """
    if symbolic_columns:
        # Define grid size for subplots
        num_columns = len(symbolic_columns)
        rows = (num_columns + 2) // 3  # Adjust rows for 3 columns per row
        cols = min(3, num_columns)
        
        fig, axs = plt.subplots(rows, cols, figsize=(cols * 5, rows * 5), squeeze=False)
        axs = axs.flatten()  # Flatten to handle subplots as a 1D array
        
        for idx, col in enumerate(symbolic_columns):
            # Count occurrences for each unique value
            counts = data[col].value_counts()
            axs[idx].bar(counts.index, counts.values)
            axs[idx].set_title(f"Histogram for {col}")
            axs[idx].set_xlabel(col)
            axs[idx].set_ylabel("Number of records")
        
        # Hide any unused subplot axes
        for ax in axs[len(symbolic_columns):]:
            ax.axis("off")
        
        plt.tight_layout()
        plt.savefig(f"images/{file_tag}_histograms_symbolic.png")
        plt.show()
    else:
        print("There are no symbolic variables.")

# Example usage:
# Load your dataset
data = pd.read_csv("dataset/class_financial distress.csv")

# Define symbolic columns (replace with actual columns in your dataset)
symbolic_columns = df.get_variable_types(data)["symbolic"]; # Example symbolic variables

# Call the function
plot_symbolic_histograms(data, symbolic_columns, file_tag="financial_distress")
