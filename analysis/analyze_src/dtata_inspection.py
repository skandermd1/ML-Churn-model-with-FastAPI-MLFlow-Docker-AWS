from abc import ABC, abstractmethod
import pandas as pd
from matplotlib import pyplot as plt



class DataInspectionStrategy(ABC):
    @abstractmethod
    def inspect(self, df: pd.DataFrame) -> None:
        """Abstract method to inspect the DataFrame."""
        pass

class DataTypeStrategy(DataInspectionStrategy):
    def inspect(self, df: pd.DataFrame) -> None:
        """Inspect the data types of the DataFrame."""
        print("numerical columns:")
        print(df.describe())
        print("\nCategorical columns:")
        print(df.describe(include=['O']))
class MissingValuesStrategy(DataInspectionStrategy):
    def inspect(self, df: pd.DataFrame) -> None:
        """Inspect the missing values in the DataFrame."""
        missing_values = df.isnull().sum()
        print("Missing Values:")
        print(missing_values[missing_values > 0]) 
class DataInspector:
    def __init__(self, strategy: DataInspectionStrategy):
        self._strategy = strategy

    def set_strategy(self, strategy: DataInspectionStrategy) -> None:
        """Set a new inspection strategy."""
        self._strategy = strategy

    def inspect(self, df: pd.DataFrame) -> None:
        """Inspect the DataFrame using the current strategy."""
        self._strategy.inspect(df)
