import os
import tensorflow as tf
import numpy as np
# from Cases_define import *
from importlib import import_module

# case = eval('Case_Permafrost')()
# case.generate_dataset(ngpu=1)
dataset_module = import_module("DefinedDataset.DatasetPermafrost")
dataset = getattr(dataset_module, "DatasetPermafrost")()
dataset.generate_dataset(ngpu=1)
dataset.animate()

batch_size = 5
sizes_inp = dataset.get_example()[0]['dispersion'].shape[:-1]
sizes_lab = dataset.get_example()[1]['vpdepth'].shape
inputs = np.empty([batch_size,*sizes_inp])
labels = np.empty([batch_size,1,*sizes_lab,1])

for i in range(batch_size):
    data = dataset.get_example()
    inputs[i] = np.abs(data[0]['dispersion'].reshape(sizes_inp))
    labels[i][0][:,0] = data[1]['vpdepth'].reshape(sizes_lab)

def build_cnn_model():
    cnn_network = tf.keras.Sequential([
        tf.keras.layers.Input(shape=inputs.shape[1:]),
        tf.keras.layers.Conv2D(filters=16, kernel_size=(3, 3), activation=tf.nn.leaky_relu, padding='same'),
        tf.keras.layers.Conv2D(filters=32, kernel_size=(3, 3), activation=tf.nn.leaky_relu, padding='same'),
        tf.keras.layers.MaxPool2D(pool_size=(2,2)),
        tf.keras.layers.Conv2D(filters=64, kernel_size=(3, 3), activation=tf.nn.leaky_relu, padding='same'),
        tf.keras.layers.Conv2D(filters=128, kernel_size=(3, 3), activation=tf.nn.leaky_relu, padding='same'),
        tf.keras.layers.MaxPool2D(pool_size=(2,2)),
        tf.keras.layers.Conv2D(filters=128, kernel_size=(3, 3), activation=tf.nn.leaky_relu, padding='same'),
        tf.keras.layers.Conv2D(filters=256, kernel_size=(3, 3), activation=tf.nn.leaky_relu, padding='same'),
        tf.keras.layers.MaxPool2D(pool_size=(2,2)),
        tf.keras.layers.Conv2D(filters=128, kernel_size=(3, 3), activation=tf.nn.leaky_relu, padding='same'),
        tf.keras.layers.Conv2D(filters=64, kernel_size=(3, 3), activation=tf.nn.leaky_relu, padding='same'),
        tf.keras.layers.MaxPool2D(pool_size=(2,2)),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(400, activation=tf.nn.softmax),
        # tf.keras.layers.Dense(800, activation=tf.nn.softmax),
        tf.keras.layers.Reshape(target_shape=(-1,400,1))
    ])
    return cnn_network

cnn_model = build_cnn_model()
cnn_model.predict(np.abs(inputs[:1]))
print(cnn_model.summary())

cnn_model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
                  loss='sparse_categorical_crossentropy',
                  # loss=tf.losses.SparseCategoricalCrossentropy,
                  metrics=['accuracy'])
cnn_model.fit(inputs, labels, batch_size=batch_size, epochs=5)


inputs_test = np.empty([batch_size,*sizes_inp])
labels_test = np.empty([batch_size,1,*sizes_lab,1])

for i in range(batch_size):
    data = dataset.get_example(phase='test')
    inputs_test[i] = np.abs(data[0]['dispersion'].reshape(sizes_inp))
    labels_test[i][0][:,0] = data[1]['vpdepth'].reshape(sizes_lab)

print("\n testing")
test_loss, tes_acc = cnn_model.evaluate(inputs_test,labels_test)
