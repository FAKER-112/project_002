from zenml.config import DockerSettings
from zenml.integrations.constants import MLFLOW
from zenml.pipelines import pipeline
from src.components.data_ingestion import initate_data_ingestion
from src.components.data_transformation import clean_data
from src.components.model_trainer import train
from src.components.evaluation import evaluation 
from zenml.integrations.mlflow.mlflow_utils import get_tracking_uri

docker_settings = DockerSettings(required_integrations= [MLFLOW])
@pipeline(enable_cache=False, settings={'docker': docker_settings})
def train_pipeline(ingest_data, clean_data, model_train, evaluation):
    '''
    Args:
        ingest_data: DataClass
        clean_data: DataClass
        model_train: DataClass
        evaluation: DataClass
    Returns:
        r2 score: float
        rmse: float
    '''
    df= ingest_data()
    x_train, x_test, y_train, y_test= clean_data(df)
    model= model_train(x_train, x_test, y_train, y_test)
    r2_score, rmse= evaluation(model, x_test, y_test)
    return r2_score, rmse


if __name__ == "__main__":
    training = train_pipeline(
        initate_data_ingestion(),
        clean_data(),
        train(),
        evaluation(),
    )

    training.run()

    print(
        "Now run \n "
        f"    mlflow ui --backend-store-uri '{get_tracking_uri()}'\n"
        "To inspect your experiment runs within the mlflow UI.\n"
        "You can find your runs tracked within the `mlflow_example_pipeline`"
        "experiment. Here you'll also be able to compare the two runs.)"
    )