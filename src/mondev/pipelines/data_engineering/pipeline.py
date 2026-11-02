from kedro.pipeline import Pipeline, node, pipeline
from .nodes import create_model_input_table, process_data, fit_scaler, tranform_scaler

def create_pipeline(**kwargs) -> Pipeline:
    return pipeline([
        node(
            func=process_data,
            inputs=["raw_data", "params:model_options"],
            outputs="processed_data",
            name="node_preprocess_healthy",
        ),
        node(
            func=process_data,
            inputs=["test_data", "params:model_options"],
            outputs="processed_test_data",
            name="node_preprocess_test",
        ),

        # Scale features
        node(fit_scaler, inputs="processed_data", outputs="scaler", name="node_fit-scaler"),
        node(tranform_scaler, inputs=["scaler", "processed_data"], outputs="scaled_data", name="node_scale-healthy"),
        node(tranform_scaler, inputs=["scaler", "processed_test_data"], outputs="scaled_test_data", name="node_scale-test"),
        node(
            func=create_model_input_table,
            inputs=["scaled_data", "params:model_options"],
            outputs="series",
            name="node_create_series",
        ),
        node(
            func=create_model_input_table,
            inputs=["scaled_test_data", "params:model_options"],
            outputs="test_series",
            name="node_create_test_series",
        )
    ])
