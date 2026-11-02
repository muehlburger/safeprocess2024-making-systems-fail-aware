import logging
from typing import Dict

from tensorflow import keras
from keras.models import Sequential
from keras.layers import Dense, LSTM, Bidirectional, Dropout, Lambda

from mondev.utils import *

log = logging.getLogger(__name__)

def model_simple_lstm(parameters):
    model = Sequential([
        Lambda(lambda x: tf.expand_dims(x, axis=-1), input_shape=[None]),
        Bidirectional(LSTM(60, return_sequences=True)),
        Dropout(0.2),
        Bidirectional(LSTM(60, return_sequences=True)),
        Dropout(0.2),
        LSTM(60),
        Dense(1),
    ], name='lstm')

    model.summary()
    return model

def train_model(train_set, valid, parameters):
    """Trains the model.

    Args:
        train_set: a tensorflow dataset.

    Returns:
        Trained model.
    """
    tf.keras.backend.clear_session()

    feature = parameters["feature"]
    model = model_simple_lstm(parameters)
    
    # Initialize the optimizer
    optimizer = tf.keras.optimizers.SGD(momentum=0.9, learning_rate=parameters["learning_rate"])

    # Set the training parameters
    model.compile(loss=tf.keras.losses.Huber(), 
                  optimizer=optimizer, 
                  metrics=["mae"])

    callbacks = [
        keras.callbacks.History(),
        tf.keras.callbacks.CSVLogger(f"data/06_models/{feature}-train_log.csv", append=False, separator=';'),
    ]

    model.fit(
        train_set,
        validation_data=valid,
        epochs=parameters["epochs"],
        callbacks=callbacks,
    )

    model.save(f"data/06_models/model_{feature}.keras")
