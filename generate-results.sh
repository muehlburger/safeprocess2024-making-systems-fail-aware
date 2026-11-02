RESULTS_FILENAME="results"

echo "Model,Data,MAE,MSE,MAPE,Window Size,Epochs,Threshold,Anomalies detected" > data/08_reporting/$RESULTS_FILENAME.csv

kedro run --pipeline=de --params="model_options.feature=volflow_l_min"
kedro run --pipeline=me --params="model_options.feature=volflow_l_min"

kedro run --pipeline=de --params="model_options.feature=amperage_A"
kedro run --pipeline=me --params="model_options.feature=amperage_A"

kedro run --pipeline=de --params="model_options.feature=temperature_degC_6"
kedro run --pipeline=me --params="model_options.feature=temperature_degC_6" 

kedro run --pipeline=de --params="model_options.feature=temperature_degC_5"
kedro run --pipeline=me --params="model_options.feature=temperature_degC_5"
kedro run --pipeline=me --params="model_options.feature=temperature_degC_5"

# different window sizes
./generate-results-temp5.sh
./generate-results-temp6.sh
./generate-results-volflow.sh
./generate-results-amperage.sh

