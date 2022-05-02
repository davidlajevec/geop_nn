import numpy as np
import tensorflow as tf
from preprocessing import data_processing, load_data
from matplotlib import pyplot as plt
import os

### CUSTOM LOSS, ACC METRIC ###

class WeightedRMSE(tf.keras.losses.Loss):
    def __init__(self):
        super().__init__()
    def call(self,y_true,y_pred):
        lat = [89.5-j*4 for j in range(45)]
        lat.insert(0, 0)
        lat.append(0)
        lat.append(0)
        lat_grid = tf.convert_to_tensor(np.expand_dims(np.array([lat for i in range(92)]).T, axis = 2), dtype=tf.float32)
        error = y_pred - y_true
        weights_lat = tf.math.cos(lat_grid*np.pi/180)
        weights_lat /= tf.reduce_mean(weights_lat)
        # tf.math.multiply fixed numerical stability
        mse = tf.reduce_mean(tf.math.multiply(tf.square(error), weights_lat))
        rmse = tf.math.sqrt(mse)
        return rmse 

def weighted_rmse(y_true, y_pred):
    lat = [89.5-j*4 for j in range(45)]
    lat.insert(0, 0)
    lat.append(0)
    lat.append(0)
    lat_grid = tf.convert_to_tensor(np.expand_dims(np.array([lat for i in range(92)]).T, axis = 2), dtype=tf.float32)
    error = y_pred - y_true
    weights_lat = tf.math.cos(lat_grid*np.pi/180)
    weights_lat /= tf.reduce_mean(weights_lat)
    # tf.math.multiply fixed numerical stability
    mse = tf.reduce_mean(tf.math.multiply(tf.square(error), weights_lat))
    rmse = tf.math.sqrt(mse)
    return rmse 

def rmse(y_true, y_pred):
    error = y_pred - y_true
    # tf.math.multiply fixed numerical stability
    mse = tf.reduce_mean(tf.square(error))
    rmse = tf.math.sqrt(mse)
    return rmse 

def accuracy(y_true, y_pred):
        
    clim = np.load('data/clima.npy', allow_pickle = True).astype('float32')

    fa = y_pred - clim
    a = y_true - clim

    lat = [89.5-j*4 for j in range(45)]
    lat.insert(0, 0)
    lat.append(0)
    lat.append(0)
    lat_grid = tf.convert_to_tensor(np.expand_dims(np.array([lat for i in range(92)]).T, axis = 2), dtype=tf.float32)
    weights_lat = tf.math.cos(lat_grid*np.pi/180)
    weights_lat /= tf.reduce_mean(weights_lat)
    w = weights_lat

    fa_prime = fa - tf.reduce_mean(fa)
    a_prime = a - tf.reduce_mean(a)

    acc = (tf.math.reduce_sum(w * fa_prime * a_prime)/
                tf.sqrt(tf.reduce_sum(w * fa_prime ** 2) * tf.reduce_sum(w * a_prime ** 2)))
    return acc

### PLOTTING ###

def plot_history(history, folderpath):
    plot_loss_hist(history, folderpath)
    plot_acc_hist(history, folderpath)

def plot_loss_hist(history, folderpath):
    plt.figure(1)
    plt.plot(history.history["loss"])
    plt.plot(history.history["val_loss"])
    plt.title('Loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.savefig(os.path.join(folderpath, "loss.png"))

def plot_weighted_rmse_hist(history, folderpath):
    plt.figure(1)
    plt.plot(history.history["weighted_rmse"])
    plt.plot(history.history["val_weighted_rmse"])
    plt.title('Weighted RMSE')
    plt.ylabel('WRMSE')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.savefig(os.path.join(folderpath, "WRMSE.png"))

def plot_acc_hist(history, folderpath):
    plt.figure(2)
    plt.plot(history.history["accuracy"])
    plt.plot(history.history["val_accuracy"])
    plt.title('Accuracy')
    plt.ylabel('acc')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.savefig(os.path.join(folderpath, "acc.png"))


### ELSE ###

def save_clima(): 
    data = data_processing(load_data())  
    clima = data.mean(axis = 0)
    clima.dump('data/clima.npy')
    return

if __name__ == '__main__':
    save_clima()
    #pass

