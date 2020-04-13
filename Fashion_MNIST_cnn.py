from functools import partial
import tensorflow as tf
from tensorflow import keras
from functools import reduce
import numpy as np

DefaultConv2D = partial(keras.layers.Conv2D,
        kernel_size=3, activation='relu', padding="SAME")

model = keras.models.Sequential([
    DefaultConv2D(filters=64, kernel_size=7, input_shape=[28, 28, 1]),
    keras.layers.MaxPooling2D(pool_size=2),
    DefaultConv2D(filters=128),
    DefaultConv2D(filters=128),
    keras.layers.MaxPooling2D(pool_size=2),
    DefaultConv2D(filters=256),
    DefaultConv2D(filters=256),
    keras.layers.MaxPooling2D(pool_size=2),
    keras.layers.Flatten(),
    keras.layers.Dense(units=128, activation='relu'),
    keras.layers.Dropout(0.5),
    keras.layers.Dense(units=64, activation='relu'),
    keras.layers.Dropout(0.5),
    keras.layers.Dense(units=10, activation='softmax'),
    ])

model.compile(loss="sparse_categorical_crossentropy",
        optimizer="sgd",
        metrics=["accuracy"])

fashion_mnist = keras.datasets.fashion_mnist
(X_train_full, y_train_full), (X_test, y_test) = fashion_mnist.load_data()   

X_valid, X_train = X_train_full[:5000] / 255.0, X_train_full[5000:] / 255.0
y_valid, y_train = y_train_full[:5000], y_train_full[5000:]

X_valid = np.expand_dims(X_valid, axis = -1)
X_train = np.expand_dims(X_train, axis = -1)

class_names = ["T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
    "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"]

checkpoint_cb = keras.callbacks.ModelCheckpoint("MNIST_cnn.h5",
        save_best_only=True)

history = model.fit(X_train, y_train, epochs=30,
        validation_data=(X_valid, y_valid), callbacks=[checkpoint_cb])