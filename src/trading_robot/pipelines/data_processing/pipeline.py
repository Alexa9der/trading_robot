from kedro.pipeline import Pipeline, node, pipeline
from trading_robot.pipelines.data_processing.nodes import nodes_load_data, nodes_inzener_features, nodes_select_features 


def create_training_and_selection_pipeline(**kwargs) -> Pipeline:
    return pipeline(
        [
            node(
                func=nodes_load_data,
                inputs=None,
                outputs="raw_data",
                name="load_data_node",
            ),
            node(
                func=nodes_inzener_features,
                inputs="raw_data",
                outputs="features",
                name="inzener_features_node",
            ),
            node(
                func=nodes_select_features,
                inputs="features",
                outputs="selected_features",
                name="select_features_node",
            ),
        ]
    )
    



    