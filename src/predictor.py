from pathlib import Path

import joblib
import pandas as pd

from src.data_processor import DataProcessor
from src.feature_engineering import MapBinarySeries


class ChurnPredictor:
    """Load the trained bundle and predict churn from raw customer records."""

    def __init__(self, model_path: str | Path):
        bundle = joblib.load(model_path)
        self.model = bundle["model"]
        self.feature_columns = bundle["feature_columns"]

    def predict(self, customers: pd.DataFrame) -> pd.DataFrame:
        if customers.empty:
            raise ValueError("customers must contain at least one row.")

        processed = DataProcessor().process_data(customers.copy())
        features = processed.drop(columns=["Churn"], errors="ignore")

        binary_columns = [
            column
            for column in features.select_dtypes(include=["object"]).columns
            if column in self.feature_columns
        ]
        for column in binary_columns:
            values = features[column].astype(str)
            if set(values.unique()) <= {"Yes", "No"}:
                features[column] = values.map({"No": 0, "Yes": 1})
            elif set(values.unique()) <= {"Male", "Female"}:
                features[column] = values.map({"Female": 0, "Male": 1})
            else:
                features[column] = MapBinarySeries().ApplyTransformation(values)

        one_hot_columns = [
            column
            for column in features.select_dtypes(include=["object"]).columns
            if any(name.startswith(f"{column}_") for name in self.feature_columns)
        ]
        if one_hot_columns:
            features = pd.get_dummies(features, columns=one_hot_columns, drop_first=True)

        features = features.reindex(columns=self.feature_columns, fill_value=0)

        predictions = self.model.predict(features).astype(int)
        probabilities = self.model.predict_proba(features)[:, 1]
        return pd.DataFrame(
            {
                "churn_prediction": predictions,
                "churn_label": pd.Series(predictions).map({0: "No", 1: "Yes"}),
                "churn_probability": probabilities,
            },
            index=customers.index,
        )