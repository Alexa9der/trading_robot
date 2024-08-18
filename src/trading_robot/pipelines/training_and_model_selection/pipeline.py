from kedro.pipeline import Pipeline, node, pipeline
from trading_robot.pipelines.training_and_model_selection.nodes import nodes_train_test_split, nodes_ml_tune
from trading_robot.pipelines.data_processing.pipeline import create_training_and_selection_pipeline


def create_model_pipeline(**kwargs) -> Pipeline:

    model_selection_pipeline = pipeline(
                                        [
                                            node(
                                                func=nodes_train_test_split,
                                                inputs="selected_features",
                                                outputs=["X_train", "y_train", "X_test", "y_test"],
                                                name="train_test_split_node",
                                            ),

                                            node(
                                                func=nodes_ml_tune,
                                                inputs=["X_train", "y_train", "X_test", "y_test"],
                                                outputs="model_indicator_mapping",
                                                name="tune_and_log_models",
                                            ),

                                        ]
                                    )
    
    data_processing_pipeline = create_training_and_selection_pipeline()

    combined_pipeline = Pipeline(
        data_processing_pipeline.nodes + model_selection_pipeline.nodes
    )

    return  combined_pipeline
    
    



    