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