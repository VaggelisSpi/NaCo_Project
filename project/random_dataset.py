import random
import pandas as pd


def create_random_dataset(df: pd.DataFrame, n: int = 100, seed: int | None = None) -> pd.DataFrame:
    """
    Get a random sample from a dataset
    The parameter n defines the size of the sample
    """
    return df.sample(n, random_state=seed, ignore_index=True)
