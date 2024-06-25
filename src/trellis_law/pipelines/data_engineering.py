from kedro.pipeline import Pipeline, node
from trellis_law.nodes.data_engineering import preprocess_data

def create_pipeline(**kwargs):
    return Pipeline(
        [
            node(
                func=preprocess_data,
                inputs=None,
                outputs="processed_data",
                name="preprocess_data_node"
            ),
        ]
    )
