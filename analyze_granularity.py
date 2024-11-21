#!/usr/bin/env python3
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from dslabs_functions import get_variable_types, plot_bar_chart, derive_date_variables, analyse_property_granularity, analyse_date_granularity, HEIGHT


# Main Program
def main():
    # Configuration
    input_file = "datasets/class_ny_arrests.csv"  # Replace with your CSV file path
    output_dir = Path("images")   # Directory to save output images
    output_dir.mkdir(exist_ok=True)
    
    # Load the dataset
    print("Loading dataset...")
    data = pd.read_csv(input_file, parse_dates=True, dayfirst=True)

    # Identify date variables
    print("Identifying date variables...")
    variables_types: dict[str, list] = get_variable_types(data)
    
    # Derive granular date variables
    print("Deriving date granularities...")
    data_ext: pd.DataFrame = derive_date_variables(data, variables_types["date"])
    
    # Analyze and save granularity histograms
    print("Analyzing and saving granularity histograms...")

    for v_date in variables_types["date"]:
        analyse_date_granularity(data, v_date, ["year", "quarter", "month", "day"])
        output_path = output_dir / f"{v_date}_granularity.png"
        plt.savefig(output_path)

    #print("loading dataset for property granularity analysis...")
    #data = pd.read_csv(input_file)

    properties = ["ARREST_PRECINCT"]
    #properties = ["Time"]
    # Analyze and save property histograms
    print("Analyzing and saving property histograms...")
    ax = analyse_property_granularity(data, "location", properties)
    output_path = output_dir / f"{properties[0]}_granularity.png"
    plt.savefig(output_path)
    plt.show()


    fig = analyse_property_granularity(data, "location", location_vars)
    output_path = output_dir / "gdindex_granularity_location.png"
    fig.savefig(output_path)
    print(f"Saved granularity chart to {output_path}")
    plt.show()


if __name__ == "__main__":
    main()
