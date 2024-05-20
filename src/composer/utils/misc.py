import pandas as pd
from tabulate import tabulate

def print_df(data: pd.DataFrame, line_count=0):
    """
    Prints the DataFrame stored in data in a tabulated format for better readability.

    Args:
        data (pd.DataFrame): The dataframe we are printing.
        line_count (int): Number of lines to print, starting from the top. If left at 0, prints all lines.
    """
    data_to_print = None
    if line_count > 0:
        data_to_print = data.head(line_count)
    else:
        data_to_print = data
    print(tabulate(data_to_print, headers='keys', tablefmt='psql', showindex="never"))
