from abc import ABC, abstractmethod
from pathlib import Path
import joblib

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
    def build_model(self, X_train: pd.DataFrame, y_train: pd.Series) -> str:
        """Builds an XGBoost model from the DataFrame."""
        # Separate features and target
        
        

        # Initialize the XGBoost classifier
        model = XGBClassifier(
            n_estimators=100,
            learning_rate=0.1,
            max_depth=3,
            random_state=42
        )

        if pd.api.types.is_numeric_dtype(y_train):
            encoded_target = y_train.astype(int)
        else:
            encoded_target = y_train.astype(str).str.strip().map({"No": 0, "Yes": 1})
        if encoded_target.isna().any():
            raise ValueError("The Churn target must contain only 'No' and 'Yes' values.")

        if set(encoded_target.unique()) - {0, 1}:
            raise ValueError("The Churn target must contain only binary values 0/1 or No/Yes.")

        project_root = Path(__file__).resolve().parent.parent
        tracking_db = project_root / "mlflow.db"
        model_path = project_root / "artifacts" / "churn_model.joblib"
        model_path.parent.mkdir(parents=True, exist_ok=True)
        mlflow.set_tracking_uri(f"sqlite:///{tracking_db.as_posix()}")
        with mlflow.start_run() :
            # Fit the model
            model.fit(X_train, encoded_target)

            mlflow.log_param("n_estimators", 100)
            mlflow.log_param("learning_rate", 0.1)
            mlflow.log_param("max_depth", 3)
            joblib.dump(
                {"model": model, "feature_columns": list(X_train.columns)},
                model_path,
            )
            mlflow.log_artifact(str(model_path), artifact_path="model")

        return str(model_path)
