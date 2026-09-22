from zenml import step
from src.data_ingestor import DataIngestorFactory
import pandas as pd
from pathlib import Path



@step(enable_cache=False)
def data_ingestion_step() -> pd.DataFrame:
    """
    Ingests data from a .zip file and returns it as a pandas DataFrame.

    Returns:
        pd.DataFrame: The ingested data.
    """
    # Resolve the archive from the repository root regardless of the launch directory.
    repository_root = Path(__file__).resolve().parent.parent
    file_path = repository_root / "data" / "archive.zip"

    # Determine the file extension
    file_extension = file_path.suffix

    # Get the appropriate DataIngestor
    data_ingestor = DataIngestorFactory.get_data_ingestor(file_extension)

    # Ingest the data and load it into a DataFrame
    df = data_ingestor.ingest(str(file_path))

    return df

