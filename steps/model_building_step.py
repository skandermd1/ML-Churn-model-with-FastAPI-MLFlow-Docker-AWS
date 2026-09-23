from zenml import step
from src.model_building import XGBoostModelBuilder
import pandas as pd

@step(enable_cache=False)
def model_building_step(X_train: pd.DataFrame, y_train: pd.Series) :
    builder = XGBoostModelBuilder()
    builder.build_model(X_train, y_train)