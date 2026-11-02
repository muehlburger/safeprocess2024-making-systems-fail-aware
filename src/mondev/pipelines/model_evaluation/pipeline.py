"""
This is a boilerplate pipeline 'model_evaluation'
generated using Kedro 0.18.3
"""

from kedro.pipeline import Pipeline, node, pipeline
from mondev.utils import *

def create_pipeline(**kwargs) -> Pipeline:
    return pipeline([
        node(
            name="node_plot_history",
            func=plot_history,
            inputs="params:model_options",
            outputs=None,
        ),
        node(
            func=predict,
            inputs=["series", "params:model_options"],
            outputs="predictions",
            name="node_simple_rnn_forecast",
        ),
        node(
            func=plot_predictions,
            inputs=["series", "predictions", "parameters", "params:filename_normal_predictions", "scaler"],
            outputs=None,
            name="node_plot_normal_predictions",
        ),
        node(
            func=plot_predictions,
            inputs=["test_series", "predictions", "parameters", "params:filename_test_vs_predictions", "scaler"],
            outputs=None,
            name="node_plot_test_vs_predictions",
        ),
        node(
            func=generate_latex_table,
            inputs="params:filename_results_csv",
            outputs=None,
            name="node_generate_latex_table",
        ),

    ])  # type: ignore

