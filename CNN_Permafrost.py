import os
import tensorflow as tf
import numpy as np
from Cases_define import *

case = eval('Case_Permafrost')()
case.generate_dataset(ngpu=1)

batch_size = 5
sizes = case.get_dimensions()
inputs = np.empty([batch_size, *sizes[0]])
labels = np.empty([batch_size, 1, *sizes[-1]])

for i in range(batch_size):
    data = case.get_example()
    inputs[i] = np.abs(data[0])
    vp_index = case.example_order.index('vp')
    labels[i][0] = data[vp_index]


def build_cnn_model():
    cnn_network = tf.keras.Sequential([
        # tf.keras.layers.Conv3D(16, [15, 1, 1], padding='same',activation=tf.nn.leaky_relu),
        # tf.keras.layers.Conv3D(16, [1, 9, 1], padding='same', activation=tf.nn.leaky_relu),
        # tf.keras.layers.Conv3D(16, [15, 1, 1], padding='same', activation=tf.nn.leaky_relu),
        # tf.keras.layers.Conv3D(16, [1, 9, 1], padding='same', activation=tf.nn.leaky_relu),
        # tf.keras.layers.Conv3D(32, [15, 1, 1], padding='same', activation=tf.nn.leaky_relu),
        # tf.keras.layers.Conv3D(32, [1, 9, 1], padding='same', activation=tf.nn.leaky_relu),
        tf.keras.layers.Conv2D(filters=16, kernel_size=(3, 3), activation=tf.nn.leaky_relu, padding='same'),
        tf.keras.layers.Conv2D(filters=32, kernel_size=(3, 3), activation=tf.nn.leaky_relu, padding='same'),
        tf.keras.layers.maxpooling()
        tf.keras.layers.Conv2D(filters=64, kernel_size=(3, 3), activation=tf.nn.leaky_relu, padding='same'),
        tf.keras.layers.Conv2D(filters=128, kernel_size=(3, 3), activation=tf.nn.leaky_relu, padding='same'),
        tf.keras.layers.maxpooling()
        tf.keras.layers.Conv2D(filters=128, kernel_size=(3, 3), activation=tf.nn.leaky_relu, padding='same'),
        tf.keras.layers.Conv2D(filters=256, kernel_size=(3, 3), activation=tf.nn.leaky_relu, padding='same'),
        tf.keras.layers.maxpooling()
        tf.keras.layers.Conv2D(filters=128, kernel_size=(3, 3), activation=tf.nn.leaky_relu, padding='same'),
        tf.keras.layers.Conv2D(filters=64, kernel_size=(3, 3), activation=tf.nn.leaky_relu, padding='same'),
        tf.keras.layers.maxpooling()
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(400, activation=tf.nn.softmax),
        tf.keras.layers.Reshape(target_shape=(-1, 400, 1))
    ])
    return cnn_network


cnn_model = build_cnn_model()
cnn_model.predict(np.abs(inputs[:1]))
print(cnn_model.summary())


cnn_model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
cnn_model.fit(inputs, labels, batch_size=batch_size, epochs=5)
