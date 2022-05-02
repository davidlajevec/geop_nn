import os
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from models import dense_model, simple_enc_dec
from preprocessing import train_valid_data
from utils import WeightedRMSE, accuracy, plot_history

def make_folder_structure(training_run_name='model'):
    foldername = 'Trained_nets'
    folderpath = os.path.join(foldername, training_run_name)
    os.makedirs(folderpath, exist_ok=True)
    return folderpath


def fit_model(model_generator, folderpath, train, val, initial_lr=1e-5,
              decay_rate=0.9, decay_steps=100, epochs=20, batch_size=32, draw_performance=True):

    model = model_generator

    with open(os.path.join(folderpath, "summary.txt"), "w+") as f:
        model.summary(print_fn=lambda x: f.write(x + '\n'))

    lr_schedule = keras.optimizers.schedules.ExponentialDecay(
        initial_lr, decay_steps=decay_steps, decay_rate=decay_rate, staircase=True)

    model.compile(
        loss=WeightedRMSE(),
        optimizer=keras.optimizers.Adam(learning_rate=lr_schedule),
        metrics=[accuracy, tf.keras.metrics.RootMeanSquaredError()],)

    checkpoint_cb = keras.callbacks.ModelCheckpoint(os.path.join(folderpath, "model.h5"), save_best_only=True)

    early_stopping_cb = keras.callbacks.EarlyStopping(monitor="val_loss", patience=20, restore_best_weights=True)

    history = model.fit(train[0],
        train[1],
        validation_data=val,
        shuffle=False,
        epochs=epochs,
        batch_size=batch_size,
        verbose=1,
        callbacks=[checkpoint_cb, early_stopping_cb],
    )

    if draw_performance:
        plot_history(history, folderpath)

    return model

def model_maker(model_name):
    model = keras.models.load_model(f"Trained_nets/{model_name}/model.h5",
        custom_objects = {'WeightedRMSE':WeightedRMSE(), 'accuracy':accuracy}, compile=False)
    model.trainable = True
    return model

if __name__ == "__main__":
    model_name, day = 'dense_b', 7
    folderpath = make_folder_structure(training_run_name=f'model_{model_name}_{day}_day')
    train_X, train_y, val_X, val_y = train_valid_data(skip_day=day, split=0.1)  

    fit_model(model_maker('model_dense_a_7_day'), folderpath, (train_X, train_y), (val_X, val_y), initial_lr=1E-5,
        decay_rate=0.95, decay_steps=365, epochs=10, batch_size=128, draw_performance=True)    

    #fit_model(dense_model, folderpath, (train_X, train_y), (val_X, val_y), initial_lr=1E-5, 
    #    decay_rate=0.95, decay_steps=365, epochs=10, batch_size=128, draw_performance=True)

  