from zenml import step
from src.data_splitter import DataSplitter,SimpleTrainTestSplitStrategy
import pandas as pd


@step(enable_cache=False)
def data_splitting_step(
    df: pd.DataFrame,
    test_size: float = 0.2,
    random_state: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """
    Split the input DataFrame into training and testing sets.

    Args:
        df (pd.DataFrame): The input DataFrame.
        test_size (float): Proportion of the dataset to include in the test split.
        random_state (int): Random seed for reproducibility.

    Returns:
        tuple: A tuple containing the training features, testing features,
               training labels, and testing labels.
    """
    splitter = DataSplitter(SimpleTrainTestSplitStrategy(test_size=test_size, random_state=random_state))
    X_train, X_test, y_train, y_test = splitter.split(df,"Churn")
    return X_train, X_test, y_train, y_test