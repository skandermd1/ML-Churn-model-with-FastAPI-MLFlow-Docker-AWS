from src.predictor import ChurnPredictor
import pandas as pd
from zenml import step
from pathlib import Path

@step
def prediction_step(data:pd.DataFrame ,model_path:str | Path) -> pd.DataFrame:
    predictor=ChurnPredictor(model_path)
    prediction=predictor.predict(data)
    return prediction
