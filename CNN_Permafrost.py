import os
import tensorflow as tf
import numpy as np
from Cases_define import *

case = eval('Case_Permafrost')()
case.generate_dataset(ngpu=1)

sizes = case.get_dimensions()
data = case.get_example()


def build_cnn_model():
    cnn_network = tf.keras.Sequential([
        tf.keras.layers.Conv2D(filters=20, kernel_size=(2, 2), activation=tf.nn.relu, padding='same'),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(20)
    ])
    return cnn_network


cnn_model = build_cnn_model()
cnn_model.predict(np.abs(data[0]))
print(cnn_model.summary())
