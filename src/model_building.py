from abc import ABC, abstractmethod
from pathlib import Path

import pandas as pd
import mlflow.xgboost
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier



class ModelBuilder(ABC):
    @abstractmethod
    def build_model(self, df: pd.DataFrame, target_col: str) :
        """Abstract method to build a model from the DataFrame."""
        pass
class XGBoostModelBuilder(ModelBuilder) :
    def build_model(self, X_train: pd.DataFrame, y_train: pd.Series) :
        """Builds an XGBoost model from the DataFrame."""
        # Separate features and target
        
        

        # Initialize the XGBoost classifier
        model = XGBClassifier(
            n_estimators=100,
            learning_rate=0.1,
            max_depth=3,
            random_state=42
        )

        encoded_target = y_train.map({"No": 0, "Yes": 1})
        if encoded_target.isna().any():
            raise ValueError("The Churn target must contain only 'No' and 'Yes' values.")

        tracking_db = Path(__file__).resolve().parent.parent / "mlflow.db"
        mlflow.set_tracking_uri(f"sqlite:///{tracking_db.as_posix()}")
        with mlflow.start_run() :
            # Fit the model
            model.fit(X_train, encoded_target)

            mlflow.log_param("n_estimators", 100)
