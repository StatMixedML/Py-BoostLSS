import pkg_resources
import pandas as pd


def load_example_data(dta_name: str) -> pd.DataFrame:
    """Returns dataframe of a sepecified simulated dataset example.
    """
    data_path = pkg_resources.resource_stream(__name__, dta_name)
    data_df = pd.read_csv(data_path)

    return data_df