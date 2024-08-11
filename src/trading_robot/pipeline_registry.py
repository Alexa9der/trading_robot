
from dotenv import load_dotenv
load_dotenv()

from typing import Dict
from kedro.pipeline import Pipeline
from trading_robot.pipelines.data_processing import pipeline as dp


def register_pipelines()-> Dict[str, Pipeline]:
    """Register the project's pipelines.

    Returns:
        A mapping from pipeline names to ``Pipeline`` objects.
    """
    data_processing_pipeline = dp.create_pipeline()

    return {
        "__default__": data_processing_pipeline,
        "dp": data_processing_pipeline,
    }



if __name__ == "__main__":
    print(register_pipelines())