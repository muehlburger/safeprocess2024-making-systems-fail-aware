RESULTS_FILENAME="results_amperage"

echo "Model,Data,MAE,MSE,MAPE,Window Size,Epochs,Threshold,Anomalies detected" > data/08_reporting/$RESULTS_FILENAME.csv

kedro run --pipeline=de --params="model_options.feature=amperage_A,model_options.window_size=60,filename_results_csv=$RESULTS_FILENAME"

# different window sizes
kedro run --pipeline=me --params="model_options.feature=amperage_A,model_options.window_size=1,filename_results_csv=$RESULTS_FILENAME"
kedro run --pipeline=me --params="model_options.feature=amperage_A,model_options.window_size=5,filename_results_csv=$RESULTS_FILENAME"
kedro run --pipeline=me --params="model_options.feature=amperage_A,model_options.window_size=10,filename_results_csv=$RESULTS_FILENAME"
kedro run --pipeline=me --params="model_options.feature=amperage_A,model_options.window_size=30,filename_results_csv=$RESULTS_FILENAME"
kedro run --pipeline=me --params="model_options.feature=amperage_A,model_options.window_size=60,filename_results_csv=$RESULTS_FILENAME"

kedro run --pipeline=me --params="model_options.feature=amperage_A,filename_results_csv=$RESULTS_FILENAME"
