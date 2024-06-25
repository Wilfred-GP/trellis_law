from kedro.pipeline import Pipeline, node
from trellis_law.nodes.deploy import save_model

def create_pipeline(**kwargs):
    return Pipeline(
        [
            node(
                func=save_model,
                inputs="model",
                outputs=None,
                name="save_model_node"
            ),
        ]
    )
