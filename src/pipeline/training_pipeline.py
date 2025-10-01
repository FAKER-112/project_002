from zenml.config import DockerSettings
from zenml.integrations.constants import MLFLOW
from zenml.pipelines import pipeline

docker_settings = DockerSettings(required_integration= [MLFLOW])
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
    x_train, x_test, y_train, y_test= clean_data()
    model= model_train(x_train, x_test, y_train, y_test)
    r2_score, rmse= evaluation(model, x_test, y_test)
    return r2_score, rmse