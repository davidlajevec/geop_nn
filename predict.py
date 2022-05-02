import tensorflow as tf
import numpy as np
from tensorflow import keras
from preprocessing import test_data
import matplotlib.pyplot as plt
from utils import rmse, accuracy, WeightedRMSE

def predict(test_X, model_name):
    trained_model = keras.models.load_model(f"Trained_nets/{model_name}/model.h5", 
        custom_objects = {'WeightedRMSE':WeightedRMSE(), 'accuracy':accuracy}, compile=False)
    y_pred = trained_model.predict(test_X)
    return y_pred

def get_test_metrics(y_true, y_pred):
    test_rmse = np.zeros(len(y_pred))
    test_acc = np.zeros(len(y_pred))
    for i in range(len(y_pred)):
        test_rmse[i] = rmse(y_true[i], y_pred[i]).numpy()
        test_acc[i] = accuracy(y_true[i], y_pred[i]).numpy()
    return test_rmse, test_acc

def iterative_predict(model_name):
    test_X, y_true = test_data(skip_day=1)
    rmse_iterative = np.zeros(8)
    accuracy_iterative = np.zeros(8)
    accuracy_iterative[0] = 1
    y_pred = predict(test_X, model_name)
    for i in range(7):
        rmse, acc = get_test_metrics(y_true, y_pred)
        rmse_iterative[i+1] = np.mean(rmse)
        accuracy_iterative[i+1] = np.mean(acc)
        if i < 6:
            y_pred = predict(y_pred, model_name)[:-2]
            _, y_true = test_data(skip_day=i+2)
    return rmse_iterative, accuracy_iterative     

def direct_predict(model_name, skip_day=1):
    test_X, y_true = test_data(skip_day=skip_day)
    y_pred = predict(test_X, model_name)
    rmse, acc = get_test_metrics(y_true, y_pred)
    return np.mean(rmse), np.mean(acc)

def direct_predict_2_channel(model_name_n, model_name_n_1, skip_day=1):
    test_X, y_true = test_data(skip_day=skip_day)
    pred_test_X = predict(test_X, model_name_n_1)
    test_X = tf.concat((test_X, pred_test_X), axis = 3)
    y_pred = predict(test_X, model_name_n)
    rmse, acc = get_test_metrics(y_true, y_pred)
    return np.mean(rmse), np.mean(acc)

def persistance():
    rmse_persistance = np.zeros(8)
    accuracy_persistance = np.zeros(8)
    for i in range(7):
        test_X, y_true = test_data(skip_day=i+1)
        rmse, acc = get_test_metrics(y_true, test_X)
        rmse_persistance[i+1] = np.mean(rmse)
        accuracy_persistance[i+1] = np.mean(acc)
    return rmse_persistance, accuracy_persistance 

def climatology():
    clim = np.load('data/clima_test.npy', allow_pickle = True).astype('float32')
    _, y_true = test_data(skip_day=1)
    rmse_clima = np.zeros(len(y_true))
    accuracy_clima = np.zeros(len(y_true))
    for i in range(len(y_true)):
        rmse_clima[i] = rmse(y_true[i], clim)
        accuracy_clima[i] = accuracy(y_true[i], clim)
    return np.mean(rmse_clima), np.mean(accuracy_clima)  

if __name__ == '__main__': 
    #rmse_iter, acc_iter = iterative_predict('model_simple_enc_dec_1_day')
    rmse_iter, acc_iter = iterative_predict('model_dense_1_day')
    rmse_per, acc_per = persistance()
    #rmse_direct, acc_direct = direct_predict('model_simple_enc_dec_7_day')
    rmse_direct, acc_direct = direct_predict('model_dense_7_day')
    rmse_clima, acc_clima = climatology()   
    with open('metrics.txt', 'w') as f:
        f.write(f'Iteration:\n  RMSE: {rmse_iter}\n  ACC: {acc_iter}\n')
        f.write(f'Persistance:\n  RMSE: {rmse_per}\n  ACC: {acc_per}\n')
        f.write(f'Direct day 1:\n  RMSE: {rmse_direct}\n  ACC: {acc_direct}\n')
        f.write(f'Climatology:\n  RMSE: {rmse_clima}\n  ACC: {acc_clima}\n')
