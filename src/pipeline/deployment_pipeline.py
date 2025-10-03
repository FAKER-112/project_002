import json
import os
import numpy as np
import pandas as pd
from src.components.data_ingestion import initate_data_ingestion
from src.components.data_transformation import clean_data
from src.components.model_trainer import train
from src.components.evaluation import evaluation 
from zenml import pipeline,step
from zenml.config import DockerSettings
from zenml.constants import DEFAULT_SERVICE_START_STOP_TIMEOUT
from zenml.integrations.constants import MLFLOW,TENSORFLOW
from zenml.integrations.mlflow.model_deployers.mlflow_model_deployer import MLFlowModelDeployer
from zenml.integrations.mlflow.steps import mlflow_model_deployer_step
from zenml.integrations.mlflow.services import MLFlowDeploymentService
from zenml.steps import BaseParameters, Output
from src.utills import get_data_for_test

dockerSettings = DockerSettings(required_integrations=[MLFLOW])

@step(enable_cache=False)
def dynamic_importer()->str:
    """Downloads the latest data from a mock API."""
    data=get_data_for_test()
    return data

class DeploymentTriggerConfiguration(BaseParameters):
    min_accuracy: float =0.9

@step
def deployment_trigger(    
    accuracy: float,
    config: DeploymentTriggerConfiguration
)->bool:
    """Implements a simple model deployment trigger that looks at the
    input model accuracy and decides if it is good enough to deploy"""
    
    return  accuracy> config.min_accuracy

class MLFlowDeploymentLoaderStepParameter(BaseParameters):
    """MLflow deployment getter parameters

    Attributes:
        pipeline_name: name of the pipeline that deployed the MLflow prediction
            server
        step_name: the name of the step that deployed the MLflow prediction
            server
        running: when this flag is set, the step only returns a running service
        model_name: the name of the model that is deployed
    """
    pipeline_name :str
    step_name : str
    running : bool =True

@step(enable_cache=False)
def prediction_service_loader(
    pipeline_name :str,
    pipeline_step_name: str,
    running : bool =True,
    model_name: str ='model'
)->MLFlowDeploymentService:
    """Get the prediction service started by the deployment pipeline.

    Args:
        pipeline_name: name of the pipeline that deployed the MLflow prediction
            server
        step_name: the name of the step that deployed the MLflow prediction
            server
        running: when this flag is set, the step only returns a running service
        model_name: the name of the model that is deployed
    """
    Model_Deployer=MLFlowModelDeployer.get_active_model_deployer()
    existing_service=Model_Deployer.find_model_server(
        pipeline_name=pipeline_name,
        pipeline_step_name= pipeline_step_name,
        model_name= model_name,
        running=running
    )
    if not existing_service:
        raise RuntimeError(
            f"No MLflow prediction service deployed by the "
            f"{pipeline_step_name} step in the {pipeline_name} "
            f"pipeline for the '{model_name}' model is currently "
            f"running."
        )
    print(existing_service)
    print(type(existing_service))
    return existing_service[0]

@step
def predictor(
    service: MLFlowDeploymentService,
    data: np.ndarray
)->np.ndarray:
    """Run an inference request against a prediction service"""

    service.start(timeout=10)  # should be a NOP if already started
    data = json.loads(data)
    data.pop("columns")
    data.pop("index")
    columns_for_df = [
        "payment_sequential",
        "payment_installments",
        "payment_value",
        "price",
        "freight_value",
        "product_name_lenght",
        "product_description_lenght",
        "product_photos_qty",
        "product_weight_g",
        "product_length_cm",
        "product_height_cm",
        "product_width_cm",
    ]
    df = pd.DataFrame(data["data"], columns=columns_for_df)
    json_list = json.loads(json.dumps(list(df.T.to_dict().values())))
    data = np.array(json_list)
    prediction = service.predict(data)
    return prediction

@pipeline(enable_cache=False, settings={'docker' : dockerSettings})
def continous_deployment_pipeline(
    minaccuracy : float=0.9,
    worker=1,
    timeout : int=DEFAULT_SERVICE_START_STOP_TIMEOUT
):
    df=initate_data_ingestion()
    x_train, x_test, y_train, y_test = clean_data(df)
    model = train(x_train, x_test, y_train, y_test)
    mse, rmse = evaluation(model, x_test, y_test)
    deployment_decision=  deployment_trigger(accuracy= mse)
    mlflow_model_deployer_step(
        model= model,
        deploy_decision= deployment_decision,
        workers=worker,
        timeout=timeout
    )
@pipeline(enable_cache=False,settings={'docker': dockerSettings})
def inference_pipeline(pipeline_name :str, pipeline_step_name:str):
    batch_data =dynamic_importer()
    model_deployment_service =prediction_service_loader(
        pipeline_name=pipeline_name,
        pipeline_step_name= pipeline_step_name,
        running=False
    )
    predictor(service= model_deployment_service, data= batch_data)