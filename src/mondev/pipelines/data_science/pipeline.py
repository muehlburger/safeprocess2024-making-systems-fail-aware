from kedro.pipeline import Pipeline, node, pipeline
from .nodes import train_model, train_test_split, windowed_dataset

def create_pipeline(**kwargs) -> Pipeline:
    return pipeline([
        node(
            name="node_train_test_split",
            func=train_test_split,
            inputs=["series", "params:model_options"],
            outputs=["x_train", "x_valid"],
        ),
        node(
            name="node_create_train_dataset",
            func=windowed_dataset,
            inputs=["x_train", "params:model_options"],
            outputs="train",
        ),
        node(
            name="node_create_valid_dataset",
            func=windowed_dataset,
            inputs=["x_valid", "params:model_options"],
            outputs="valid",
        ),
        node(
            name="node_train_model",
            func=train_model,
            inputs=["train", "valid", "params:model_options"],
            outputs=None,
        ),
    ])
