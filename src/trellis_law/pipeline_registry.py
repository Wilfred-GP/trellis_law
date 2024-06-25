from .pipelines.data_engineering import create_pipeline as create_data_engineering_pipeline
from .pipelines.model_training import create_pipeline as create_model_training_pipeline
from .pipelines.deploy import create_pipeline as create_deploy_pipeline

def register_pipelines():
    data_engineering_pipeline = create_data_engineering_pipeline()
    model_training_pipeline = create_model_training_pipeline()
    deploy_pipeline = create_deploy_pipeline()

    return {
        "de": data_engineering_pipeline,
        "mt": model_training_pipeline,
        "dp": deploy_pipeline,
        "__default__": data_engineering_pipeline + model_training_pipeline + deploy_pipeline
    }
