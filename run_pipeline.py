from zenml.integrations.mlflow.mlflow_utils import get_tracking_uri
import click
from steps.data_ingestion_step import data_ingestion_step
from steps.data_processing_step import data_processing_step
from steps.feature_engineering_step import feature_engineering_step
from steps.data_splitting_step import data_splitting_step
from steps.model_building_step import model_building_step
import pandas as pd


def main():

    df=data_ingestion_step()
    print("Data Ingestion Step Completed. DataFrame shape:", df.shape)
    df1=data_processing_step(df)
    print("Data Processing Step Completed. DataFrame shape:", df1.shape)
    df2=feature_engineering_step(df1)
    print("Feature Engineering Step Completed. DataFrame shape:", df2.shape)
    X_train, X_test, y_train, y_test = data_splitting_step(df2, test_size=0.2, random_state=42)
    print("Data Splitting Step Completed.")
    model_building_step(X_train, y_train)
    print("Model Building Step Completed.")









if __name__ == "__main__":
    main()
