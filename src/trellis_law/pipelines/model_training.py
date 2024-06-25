from kedro.pipeline import Pipeline, node
from trellis_law.nodes.model_training import train_models

def create_pipeline(**kwargs):
    return Pipeline(
        [
            node(
                func=train_models,
                inputs="processed_data",
                outputs="model",
                name="train_models_node"
            ),
        ]
    )
