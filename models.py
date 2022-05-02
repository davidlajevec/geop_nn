from matplotlib import units
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from utils import rmse, accuracy, WeightedRMSE

def simple_enc_dec(width=92, height=48, alpha=1.0):
    
    inputs = keras.Input((height, width, 1))

    # DOBRE IZBIRE ZA KERNEL 4 ALI 6
    x = layers.Conv2D(filters=32, kernel_size=4, padding='same', strides=1, kernel_initializer='he_normal')(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.ELU(alpha=alpha)(x)
    x = layers.MaxPool2D(pool_size=(2,2), padding='same')(x)

    x = layers.Conv2D(filters=32, kernel_size=4, padding='same', strides=1, kernel_initializer='he_normal')(x)
    x = layers.BatchNormalization()(x)
    x = layers.ELU(alpha = alpha)(x)
    x = layers.MaxPool2D(pool_size=(2,2), padding='same')(x)

    x = layers.Conv2D(filters=32, kernel_size=4, padding='same', strides=1, kernel_initializer='he_normal')(x)
    x = layers.BatchNormalization()(x)
    x = layers.ELU(alpha = alpha)(x) 

    x = layers.Conv2D(filters=64, kernel_size=4, padding='same', strides=1, kernel_initializer='he_normal')(x)
    x = layers.BatchNormalization()(x)
    x = layers.ELU(alpha = alpha)(x) 

    x = layers.Conv2D(filters=32, kernel_size=4, padding='same', strides=1, kernel_initializer='he_normal')(x)
    x = layers.BatchNormalization()(x)
    x = layers.ELU(alpha = alpha)(x) 

    x = layers.UpSampling2D(size = (2, 2))(x)
    x = layers.Conv2D(filters=32, kernel_size=4, padding='same', strides=1, kernel_initializer='he_normal')(x)
    x = layers.BatchNormalization()(x)
    x = layers.ELU(alpha = alpha)(x)

    x = layers.UpSampling2D(size = (2, 2))(x)
    x = layers.Conv2D(filters=1, kernel_size=4, padding='same', strides=1, kernel_initializer='he_normal')(x)
    x = layers.BatchNormalization()(x)

    outputs = layers.ELU(alpha = alpha)(x)

    model = keras.Model(inputs, outputs, name="2Dencoderdecoder")

    return model

def dense_model(width=92, height=48, alpha=1.0):
    inputs = keras.Input((height, width, 1))

    x = layers.Flatten()(inputs)
    x = layers.Dense(units=4416, activation="selu", kernel_initializer=tf.keras.initializers.LecunUniform())(x)
    x = layers.Dense(units=4416, activation="selu", kernel_initializer=tf.keras.initializers.LecunUniform())(x)
    x = layers.Dense(units=2000, activation="selu", kernel_initializer=tf.keras.initializers.LecunUniform())(x)
    x = layers.Dense(units=500, activation="selu", kernel_initializer=tf.keras.initializers.LecunUniform())(x)
    x = layers.Dense(units=2000, activation="selu", kernel_initializer=tf.keras.initializers.LecunUniform())(x)
    x = layers.Dense(units=4416, activation="selu", kernel_initializer=tf.keras.initializers.LecunUniform())(x)
    x = layers.Dense(units=4416, activation="selu", kernel_initializer=tf.keras.initializers.LecunUniform())(x)

    outputs = layers.Reshape((48,92,1))(x)

    model = keras.Model(inputs, outputs, name="2Dencoderdecoder")

    return model

if __name__=='__main__':
    model = dense_model()
    print(model.summary())