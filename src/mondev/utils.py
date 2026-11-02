import matplotlib.pyplot as plt
import tensorflow as tf
import pandas as pd
import numpy as np
import logging
import seaborn as sns
from pathlib import Path

log = logging.getLogger(__name__)

REPORTING_PATH = Path() / "data"/ "08_reporting"

def set_size(width, fraction=1, subplots=(1, 1)):
    """Set figure dimensions to avoid scaling in LaTeX.

    Parameters
    ----------
    width: float or string
            Document width in points, or string of predined document type
    fraction: float, optional
            Fraction of the width which you wish the figure to occupy
    subplots: array-like, optional
            The number of rows and columns of subplots.
    Returns
    -------
    fig_dim: tuple
            Dimensions of figure in inches
    """
    if width == 'thesis':
        width_pt = 426.79135
    elif width == 'beamer':
        width_pt = 307.28987
    else:
        width_pt = width

    # Width of figure (in pts)
    fig_width_pt = width_pt * fraction
    # Convert from pt to inches
    inches_per_pt = 1 / 72.27

    # Golden ratio to set aesthetic figure height
    # https://disq.us/p/2940ij3
    golden_ratio = (5**.5 - 1) / 2

    # Figure width in inches
    fig_width_in = fig_width_pt * inches_per_pt
    # Figure height in inches
    fig_height_in = fig_width_in * golden_ratio * (subplots[0] / subplots[1])

    return (fig_width_in, fig_height_in)

def save_fig(fig_id, tight_layout=True, fig_extension="pdf", resolution=1200):
    path = REPORTING_PATH / f"{fig_id}.{fig_extension}"
    if tight_layout:
        plt.tight_layout()
    plt.savefig(path, format=fig_extension, dpi=resolution, bbox_inches='tight')

def save_results(report, filename):
    path = REPORTING_PATH / f"{filename}.csv"
    log.info(f"Saving results to {path}")
    report.to_csv(path, index=False,mode='a', header=False)
    df = pd.read_csv(path)
    df = df.drop_duplicates()
    df = df.sort_values(by=['Model', 'Data'])
    df.to_csv(path, index=False)

def generate_latex_table(filename):
    """Generates a latex table with the results of the models"""
    path = REPORTING_PATH / f"{filename}.csv"
    df = pd.read_csv(path)
    df = df.round(2)
    df = df.drop_duplicates()
    df = df.sort_values(by=['Model', 'Data'])
    latex = df.to_latex(index=False, escape=False, float_format="{:.2f}".format)
    latex = latex.replace("toprule", "hline")
    latex = latex.replace("midrule", "hline")
    latex = latex.replace("bottomrule", "hline")
    latex = latex.replace("llrrrrr", "llccccc")
    latex = latex.replace("temperature_degC_5", "temp5")
    latex = latex.replace("temperature_degC_6", "temp6")
    latex = latex.replace("volflow_l_min", "volflow")
    latex = latex.replace("amperage_A", "amperage")
    latex = latex.replace("test_predictions", "test")
    latex = latex.replace("train_predictions", "train")
    latex = latex.replace("_", "\_")
    latex = latex.replace("#", "\#")

    
    path = REPORTING_PATH / f"{filename}_table.tex"
    with open(path, "w") as f:
        f.write(latex)

def plot_history(parameters):
    """Plots the history of the model.

    Args:
        parameters: the parameters of the model.

    Returns:
        Plot of the history.
    """
    feature = parameters["feature"]
    history = pd.read_csv(f"data/06_models/{feature}-train_log.csv", sep=";")

    fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True)
    fig.set_size_inches(8, 6)
    fig.set_label(f"Feature {feature}")
    ax1.set_title(f"Model loss ({feature})")
    ax1.set_ylabel('Loss')
    ax1.plot(history['loss'], label='Train')
    ax1.plot(history['val_loss'], label='Validation')
    ax1.legend()
    ax2.set_title(f"Mean Absolute Error ({feature})")
    ax2.set_ylabel('MAE')
    ax2.set_xlabel('Epoch')
    ax2.plot(history['mae'], label='Train')
    ax2.plot(history['val_mae'], label='Validation')
    ax2.legend()

    save_fig(f"history_{feature}")

def plot_predictions(series, preds, parameters, filename, scaler):
    model_options = parameters["model_options"]
    width = parameters["width"]
    feature = model_options["feature"]
    epochs = model_options["epochs"]
    threshold_train = model_options[f"threshold_{feature}"]
    window_size = model_options["window_size"]

    # TODO use the threshold from the csv file
    # if(filename.startswith("test")):
    #   thresholds = pd.read_csv(f"data/08_reporting/thresholds.csv", sep=";", header=0)
    #   log.info(f"columns: {thresholds.columns}")
    #   log.info(f"thresholds: {thresholds}")
    #   threshold = thresholds.loc[thresholds['Model'] == feature]
    #   log.info(f"found threshold: {threshold}")
                           
    
    plt.style.use('tableau-colorblind10')
    plt.rcParams.update({
            "text.usetex": True,
            "font.family": "serif",
            "font.size": 14,
            # Make the legend/label fonts a little smaller
            "legend.fontsize": 12,
            "axes.labelsize": 12,
            "axes.titlesize": 12,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
            "xtick.direction": "inout",
            "ytick.direction": "inout",
            "legend.handlelength": 1
    })

    series = scaler.inverse_transform(series)
    preds = scaler.inverse_transform(preds)
#    pred_index = 3941

    series = series[window_size - 1:]

    # log.info("preds.shape: %s", preds.shape)
    # log.info("series.shape: %s", series.shape)
    # log.info(f"series[{pred_index - 1}]: {series[pred_index - 1]}")
    # log.info(f"preds[{pred_index - 1}]: {preds[pred_index - 1]}")

    # log.info(f"series[{pred_index}]: {series[pred_index]}")
    # log.info(f"preds[{pred_index}]: {preds[pred_index]}")

    # log.info(f"series[{pred_index + 1}]: {series[pred_index + 1]}")
    # log.info(f"preds[{pred_index + 1}]: {preds[pred_index + 1]}")

    losses = tf.keras.losses.mean_absolute_error(series, preds)
    threshold_calculated = calculate_threshold(losses)
    log.info(f"threshold_calculated for {feature} ({filename}): {threshold_calculated}")

    mae_result = get_error(series, preds, "mae")
    mape_result = get_error(series, preds, "mape")
    mse_result = get_error(series, preds, "mse")
    log.info(f"Mean Absolute Error (MAE) {feature}/{filename}: {mae_result}")
    log.info(f"Mean Absolute Percentage Error (MAPE) {feature}/{filename}: {mape_result}")
    log.info(f"Mean Squared Error (MSE) {feature}/{filename}: {mse_result}")

    series = series.squeeze()
    preds = preds.squeeze()

    fig, ax1 = plt.subplots(1, 1, figsize=set_size(width))
    ax1.plot(series, linewidth=0.5, label='Actual')
    ax1.plot(preds, '--', linewidth=0.5, label='Predicted')
    ax1.set_title(f"{ filename }, mean absolute error: {mae_result}")
    ax1.fill_between(np.arange(len(series)), series, preds, color="lightcoral", alpha=0.9, label="Error")
    fig.tight_layout()
    plt.legend()
    
    save_fig(f"{feature}_{filename}")
  
    fig = sns.displot(losses, bins=50, kde=True)
    fig.set(xlabel="Mean Absolute Error Loss", ylabel="Frequency")
    fig.set(title=f"{filename}, max error: {np.max(losses)}")

    save_fig(f"{feature}_{filename}_displot")
    
    df = pd.DataFrame(data=series)
    df['loss'] = losses
    df['threshold'] = threshold_train
    df['anomaly'] = df.loss > df.threshold
    df['series'] = series

    anomalies = df.loc[df.anomaly == True]
    log.info(f"# of anomalies: {len(anomalies)}")
    
    data = {
      "Model": [feature],
      "Data": [filename],
      "MAE": [mae_result],
      "MSE": [mse_result],
      "MAPE": [mape_result],
      "Window size": [window_size],
      "Epochs": epochs,
      "Threshold": [threshold_train],
      "Anomalies": [len(anomalies)],
    }

    # Create a DataFrame
    report = pd.DataFrame(data)
    log.info(report)

    reports_filename = parameters["filename_results_csv"]
    log.info(f"reports_filename: {reports_filename}")
    save_results(report, reports_filename)

    if(filename.startswith("train")):
      save_thresholds_to_csv_dataset(threshold_train, filename, feature, window_size, epochs)

    fig, ax1 = plt.subplots(1, 1)
    ax1.plot(df.series, label='Normal')
    ax1.set_title(f"Anomalies with threshold: {threshold_train}")
    ax1.scatter(anomalies.index, anomalies.series, color='red', s=5, label='Anomaly')
    fig.tight_layout()
    plt.legend()

    save_fig(f"{feature}_{filename}_anomalies")

    #plot_series(np.arange(len(series), dtype="float32"), (series, preds), end=len(preds))

def save_thresholds_to_csv_dataset(threshold, filename, feature, window_size, epochs):
    """Saves the threshold to the dataset"""
    path = REPORTING_PATH / "thresholds.csv"
    df = pd.DataFrame(data={
      "Model": [feature], 
      "Data": [filename], 
      "Window size": [window_size], 
      "Epochs": [epochs],
      "Threshold": [threshold],  
    })
    df.to_csv(path, index=False,mode='a', header=not path.exists())
    df = pd.read_csv(path)
    df = df.drop_duplicates()
    df = df.sort_values(by=['Model', 'Data'])
    df.to_csv(path, index=False)

def windowed_dataset(dataset, params):
    """Generates dataset windows

    Args:
      series (array of float) - contains the values of the time series
      n_steps (int) - the number of time steps to include in the feature
      batch_size (int) - the batch size
      shuffle_buffer(int) - buffer size to use for the shuffle method

    Returns:
      dataset (TF Dataset) - TF Dataset containing time windows
    """
    window_size = params['window_size']
    batch_size = params['batch_size']
    shuffle_buffer_size = params['shuffle_buffer_size']
       
    # Window the data but only take those with the specified size
    dataset = dataset.window(window_size + 1, shift=1, drop_remainder=True)
    
    # Flatten the windows by putting its elements in a single batch
    dataset = dataset.flat_map(lambda window: window.batch(window_size + 1))

    # Create tuples with features and labels 
    dataset = dataset.map(lambda window: (window[:-1], window[-1]))

    # Shuffle the windows
    dataset = dataset.shuffle(shuffle_buffer_size)
    
    # Create batches of windows
    dataset = dataset.batch(batch_size).prefetch(1)

    for example_inputs, example_labels in dataset.take(1):
      log.info(f'Inputs shape (batch, time, features): {example_inputs.shape}')
      log.info(f'Labels shape (batch, time, features): {example_labels.shape}')

    return dataset

def train_test_split(series, params):
    """Generates train and validation datasets

    Args:
      series (array of float) - contains the values of the time series
      params (dict) - dictionary containing the parameters for the model

    Returns:
      x_train (TF Dataset) - TF Dataset containing training time windows
      x_valid (TF Dataset) - TF Dataset containing validation time windows
    """
    train_size = params['train_size']
    
    dataset = tf.data.Dataset.from_tensor_slices(series)
    dataset_length = dataset.cardinality().numpy()

    log.info(f'length of dataset: {dataset_length}')
    
    split_index = int(dataset_length * train_size)
    log.info(f'split_index: {split_index}')

    x_train = dataset.take(split_index)
    x_valid = dataset.skip(split_index)

    log.info(f'length of train dataset: {x_train.cardinality().numpy()}')
    log.info(f'length of valid dataset: {x_valid.cardinality().numpy()}')

    return x_train, x_valid

def predict(series, parameters):
    """Uses an input model to generate predictions on windows of data

    Args:
      model (TF Keras Model) - model that accepts data windows
      series (array of float) - contains the values of the time series
      window_size (int) - the number of time steps to include in the window
      batch_size (int) - the batch size
      features (list of str) - list of features to use in the model
      scaler(scaler) - the scaler used to scale the data (for inverse transform

    Returns:
      forecast (pandas dataframe) - dataframe containing predictions
    """
    window_size = parameters["window_size"]
    batch_size = parameters["batch_size"]
    feature = parameters["feature"]
    log.info(f"feature: {feature}")

    model = tf.keras.models.load_model(f"data/06_models/model_{feature}.keras", safe_mode=False)

    dataset = tf.data.Dataset.from_tensor_slices(series)
    dataset = dataset.window(window_size, shift=1, drop_remainder=True)
    dataset = dataset.flat_map(lambda window: window.batch(window_size))
    dataset = dataset.batch(batch_size).prefetch(1)
    predictions = model.predict(dataset)
    predictions = pd.DataFrame(data=predictions, index=series.index[window_size-1:], columns=series.columns)

    return predictions

def get_error(series, preds, error_type="mae"):
    """Calculates the error for the given series and predictions"""
    if error_type == "mae":
      error = tf.keras.metrics.MeanAbsoluteError()
    if error_type == "mape":
      error = tf.keras.metrics.MeanAbsolutePercentageError()
    if error_type == "mse":
      error = tf.keras.metrics.MeanSquaredError()


    error.update_state(series, preds)
    return error.result().numpy() 

def calculate_threshold(losses):
    """Calculates the threshold for the loss function

    Args:
      losses (array of float) - contains the loss values

    Returns:
      threshold (float) - the threshold value
    """
    #threshold = np.mean(losses) + np.std(losses)
    threshold = np.max(losses)
    return threshold
