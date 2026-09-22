from zenml import step
from src.data_processor import DataProcessor
import pandas as pd


@step(enable_cache=False)
def data_processing_step(df) -> pd.DataFrame:
    """
    Process the ingested data using the DataProcessor class.

    Args:
        df (pd.DataFrame): The ingested data.

    Returns:
        pd.DataFrame: The processed data.
    """
    processor = DataProcessor()
    processed_df = processor.process_data(df)
    return processed_df
