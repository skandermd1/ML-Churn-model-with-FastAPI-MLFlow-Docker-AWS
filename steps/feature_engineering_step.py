from zenml import step
from src.feature_engineering import BuildFeatures
import pandas as pd

@step(enable_cache=False)
def feature_engineering_step(df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply feature engineering to the input DataFrame.

    Args:
        df (pd.DataFrame): The input DataFrame.

    Returns:
        pd.DataFrame: The DataFrame after feature engineering.
    """
    feature_builder = BuildFeatures()
    processed_df = feature_builder.ApplyTransformation(df)
    return processed_df