from typing import Dict
from kedro.pipeline import Pipeline
from trading_robot.pipelines.data_processing import pipeline as dp
from trading_robot.pipelines.training_and_model_selection import pipeline as mp



def register_pipelines()-> Dict[str, Pipeline]:
    """Register the project's pipelines.

    Returns:
        A mapping from pipeline names to ``Pipeline`` objects.
    """
    data_processing_pipeline = dp.create_training_and_selection_pipeline()
    model_selection_pipeline = mp.create_model_pipeline() 


    return {
            "__default__": data_processing_pipeline + model_selection_pipeline,  # Объедините в один пайплайн по умолчанию
            "data_processing_pipeline": data_processing_pipeline,
            "model_selection_pipeline": model_selection_pipeline,
        }



if __name__ == "__main__":
    print(register_pipelines())